"""
WASM Pipeline Automation for Omniverse Integration
Connects Rust transpiler output to Omniverse runtime

Week 28 - DGX Cloud Integration & Omniverse WASM Integration
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import boto3
import redis
import requests
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "extension"))

from wasm_bridge.wasmtime_bridge import WasmtimeBridge, WasmModuleConfig


@dataclass
class WasmArtifact:
    """WASM artifact metadata"""
    artifact_id: str
    source_python_file: str
    rust_translation_path: Path
    wasm_binary_path: Path
    created_at: datetime
    size_bytes: int
    optimization_level: str
    target_functions: List[str]
    metadata: Dict[str, Any]


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # Storage
    s3_bucket: str = "portalis-wasm-artifacts"
    s3_prefix: str = "omniverse/"
    redis_url: str = "redis://localhost:6379"

    # DGX Cloud
    transpiler_service_url: str = "http://portalis-transpiler:8080"
    orchestration_service_url: str = "http://portalis-orchestration:8081"

    # Omniverse
    omniverse_extension_path: Path = Path("/opt/omniverse/extensions/portalis.wasm.runtime")
    wasm_module_cache: Path = Path("/mnt/cache/wasm_modules")

    # Monitoring
    prometheus_pushgateway: str = "http://prometheus-pushgateway:9091"

    # Pipeline settings
    auto_deploy: bool = True
    validate_wasm: bool = True
    enable_caching: bool = True
    max_concurrent_deployments: int = 10


class WasmPipelineAutomation:
    """
    Automates end-to-end WASM deployment pipeline

    Flow:
    1. Python source → Rust translation (DGX Cloud)
    2. Rust → WASM compilation
    3. WASM artifact storage (S3)
    4. Omniverse deployment
    5. Real-time monitoring
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize clients
        self.s3_client = boto3.client('s3')
        self.redis_client = redis.from_url(self.config.redis_url)
        self.wasm_bridge = None  # Initialized on demand

        # Metrics
        self.total_deployments = 0
        self.successful_deployments = 0
        self.failed_deployments = 0

        self.logger.info("WASM Pipeline Automation initialized")

    async def translate_python_to_rust(
        self,
        python_file: Path,
        output_dir: Path,
        optimization_level: str = "release"
    ) -> Path:
        """
        Submit Python file to DGX transpiler service

        Args:
            python_file: Path to Python source file
            output_dir: Directory for Rust output
            optimization_level: Optimization level (debug, release)

        Returns:
            Path to generated Rust file
        """
        self.logger.info(f"Translating {python_file} to Rust...")

        with open(python_file, 'r') as f:
            python_source = f.read()

        # Submit to transpiler service
        response = requests.post(
            f"{self.config.transpiler_service_url}/api/v1/translate",
            json={
                "source": python_source,
                "source_path": str(python_file),
                "optimization_level": optimization_level,
                "target": "wasm"
            },
            timeout=300
        )

        if response.status_code != 200:
            raise RuntimeError(f"Translation failed: {response.text}")

        result = response.json()

        # Save Rust translation
        rust_file = output_dir / f"{python_file.stem}.rs"
        rust_file.parent.mkdir(parents=True, exist_ok=True)

        with open(rust_file, 'w') as f:
            f.write(result['rust_code'])

        # Save Cargo.toml if provided
        if 'cargo_toml' in result:
            cargo_toml = output_dir / "Cargo.toml"
            with open(cargo_toml, 'w') as f:
                f.write(result['cargo_toml'])

        self.logger.info(f"Translation complete: {rust_file}")

        return rust_file

    async def compile_rust_to_wasm(
        self,
        rust_file: Path,
        output_path: Path,
        optimization: str = "release"
    ) -> Path:
        """
        Compile Rust to WASM using cargo

        Args:
            rust_file: Path to Rust source
            output_path: Output path for WASM binary
            optimization: Build profile (debug, release)

        Returns:
            Path to WASM binary
        """
        self.logger.info(f"Compiling {rust_file} to WASM...")

        # Ensure we have Cargo.toml in the directory
        cargo_dir = rust_file.parent

        # Run cargo build
        import subprocess

        cmd = [
            "cargo", "build",
            f"--{optimization}",
            "--target", "wasm32-unknown-unknown",
            "--manifest-path", str(cargo_dir / "Cargo.toml")
        ]

        result = subprocess.run(
            cmd,
            cwd=cargo_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed: {result.stderr}")

        # Find the output WASM file
        profile = "release" if optimization == "release" else "debug"
        wasm_file = cargo_dir / "target" / "wasm32-unknown-unknown" / profile / f"{rust_file.stem}.wasm"

        if not wasm_file.exists():
            # Try to find it in target directory
            wasm_files = list((cargo_dir / "target" / "wasm32-unknown-unknown" / profile).glob("*.wasm"))
            if wasm_files:
                wasm_file = wasm_files[0]
            else:
                raise FileNotFoundError(f"WASM binary not found after compilation")

        # Copy to output path
        import shutil
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wasm_file, output_path)

        self.logger.info(f"Compilation complete: {output_path} ({output_path.stat().st_size} bytes)")

        return output_path

    async def validate_wasm(self, wasm_path: Path) -> Dict[str, Any]:
        """
        Validate WASM binary

        Args:
            wasm_path: Path to WASM file

        Returns:
            Validation results
        """
        self.logger.info(f"Validating WASM: {wasm_path}")

        # Initialize bridge if needed
        if not self.wasm_bridge:
            config = WasmModuleConfig(
                cache_dir=self.config.wasm_module_cache,
                enable_validation=True
            )
            self.wasm_bridge = WasmtimeBridge(config)

        try:
            # Try to load module
            module_id = self.wasm_bridge.load_module(wasm_path)

            # Get module info
            info = self.wasm_bridge.get_module_info(module_id)

            # Unload module
            self.wasm_bridge.unload_module(module_id)

            return {
                "valid": True,
                "exports": info['exports'],
                "size_bytes": wasm_path.stat().st_size,
                "error": None
            }

        except Exception as e:
            return {
                "valid": False,
                "exports": [],
                "size_bytes": wasm_path.stat().st_size,
                "error": str(e)
            }

    async def upload_to_s3(
        self,
        local_path: Path,
        s3_key: str,
        metadata: Dict[str, str] = None
    ) -> str:
        """
        Upload WASM artifact to S3

        Args:
            local_path: Local file path
            s3_key: S3 object key
            metadata: Optional metadata

        Returns:
            S3 URI
        """
        self.logger.info(f"Uploading {local_path} to s3://{self.config.s3_bucket}/{s3_key}")

        extra_args = {}
        if metadata:
            extra_args['Metadata'] = metadata

        self.s3_client.upload_file(
            str(local_path),
            self.config.s3_bucket,
            s3_key,
            ExtraArgs=extra_args
        )

        s3_uri = f"s3://{self.config.s3_bucket}/{s3_key}"

        self.logger.info(f"Upload complete: {s3_uri}")

        return s3_uri

    async def cache_artifact_metadata(
        self,
        artifact: WasmArtifact,
        ttl_seconds: int = 86400
    ):
        """
        Cache artifact metadata in Redis

        Args:
            artifact: WASM artifact
            ttl_seconds: Cache TTL
        """
        key = f"wasm:artifact:{artifact.artifact_id}"

        data = {
            "artifact_id": artifact.artifact_id,
            "source_python_file": artifact.source_python_file,
            "rust_translation_path": str(artifact.rust_translation_path),
            "wasm_binary_path": str(artifact.wasm_binary_path),
            "created_at": artifact.created_at.isoformat(),
            "size_bytes": artifact.size_bytes,
            "optimization_level": artifact.optimization_level,
            "target_functions": artifact.target_functions,
            "metadata": artifact.metadata
        }

        self.redis_client.setex(
            key,
            ttl_seconds,
            json.dumps(data)
        )

        self.logger.debug(f"Cached artifact metadata: {key}")

    async def deploy_to_omniverse(
        self,
        wasm_path: Path,
        module_name: str,
        deployment_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Deploy WASM module to Omniverse

        Args:
            wasm_path: Path to WASM binary
            module_name: Module name for Omniverse
            deployment_config: Optional deployment configuration

        Returns:
            Deployment result
        """
        self.logger.info(f"Deploying {wasm_path} to Omniverse as '{module_name}'")

        deployment_config = deployment_config or {}

        # Copy WASM to Omniverse extension directory
        omniverse_wasm_dir = self.config.omniverse_extension_path / "wasm_modules"
        omniverse_wasm_dir.mkdir(parents=True, exist_ok=True)

        dest_path = omniverse_wasm_dir / f"{module_name}.wasm"

        import shutil
        shutil.copy2(wasm_path, dest_path)

        # Create deployment manifest
        manifest = {
            "module_name": module_name,
            "wasm_path": str(dest_path),
            "deployed_at": datetime.now().isoformat(),
            "source_size_bytes": wasm_path.stat().st_size,
            "config": deployment_config
        }

        manifest_path = omniverse_wasm_dir / f"{module_name}.manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(f"Deployment complete: {dest_path}")

        return {
            "success": True,
            "wasm_path": str(dest_path),
            "manifest_path": str(manifest_path),
            "module_name": module_name
        }

    async def publish_metrics(
        self,
        metric_name: str,
        value: float,
        labels: Dict[str, str] = None
    ):
        """
        Publish metrics to Prometheus Pushgateway

        Args:
            metric_name: Metric name
            value: Metric value
            labels: Optional labels
        """
        labels = labels or {}

        # Format for Prometheus
        label_str = ','.join([f'{k}="{v}"' for k, v in labels.items()])
        metric_line = f'{metric_name}{{{label_str}}} {value}'

        try:
            requests.post(
                f"{self.config.prometheus_pushgateway}/metrics/job/wasm_pipeline",
                data=metric_line,
                timeout=5
            )
        except Exception as e:
            self.logger.warning(f"Failed to publish metrics: {e}")

    async def process_pipeline(
        self,
        python_file: Path,
        module_name: str = None,
        auto_deploy: bool = None
    ) -> WasmArtifact:
        """
        Execute end-to-end pipeline

        Args:
            python_file: Source Python file
            module_name: Module name (defaults to file stem)
            auto_deploy: Whether to auto-deploy to Omniverse

        Returns:
            WASM artifact
        """
        start_time = datetime.now()

        module_name = module_name or python_file.stem
        auto_deploy = auto_deploy if auto_deploy is not None else self.config.auto_deploy

        self.logger.info(f"Processing pipeline for {python_file} → {module_name}")

        # Create artifact ID
        artifact_id = f"{module_name}_{start_time.strftime('%Y%m%d_%H%M%S')}"

        # Working directory
        work_dir = self.config.wasm_module_cache / artifact_id
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Python → Rust translation
            rust_file = await self.translate_python_to_rust(
                python_file,
                work_dir,
                optimization_level="release"
            )

            # Step 2: Rust → WASM compilation
            wasm_file = await self.compile_rust_to_wasm(
                rust_file,
                work_dir / f"{module_name}.wasm",
                optimization="release"
            )

            # Step 3: Validate WASM
            if self.config.validate_wasm:
                validation = await self.validate_wasm(wasm_file)
                if not validation['valid']:
                    raise RuntimeError(f"WASM validation failed: {validation['error']}")

            # Step 4: Upload to S3
            s3_key = f"{self.config.s3_prefix}{artifact_id}/{module_name}.wasm"
            s3_uri = await self.upload_to_s3(
                wasm_file,
                s3_key,
                metadata={
                    "module_name": module_name,
                    "source_file": str(python_file),
                    "created_at": start_time.isoformat()
                }
            )

            # Create artifact metadata
            artifact = WasmArtifact(
                artifact_id=artifact_id,
                source_python_file=str(python_file),
                rust_translation_path=rust_file,
                wasm_binary_path=wasm_file,
                created_at=start_time,
                size_bytes=wasm_file.stat().st_size,
                optimization_level="release",
                target_functions=[],  # TODO: Extract from Rust
                metadata={
                    "s3_uri": s3_uri,
                    "s3_key": s3_key
                }
            )

            # Step 5: Cache metadata
            if self.config.enable_caching:
                await self.cache_artifact_metadata(artifact)

            # Step 6: Deploy to Omniverse (if enabled)
            if auto_deploy:
                deployment = await self.deploy_to_omniverse(
                    wasm_file,
                    module_name
                )
                artifact.metadata['omniverse_deployment'] = deployment

            # Step 7: Publish metrics
            duration_seconds = (datetime.now() - start_time).total_seconds()

            await self.publish_metrics(
                "portalis_wasm_pipeline_duration_seconds",
                duration_seconds,
                labels={"module": module_name, "status": "success"}
            )

            await self.publish_metrics(
                "portalis_wasm_artifact_size_bytes",
                artifact.size_bytes,
                labels={"module": module_name}
            )

            self.total_deployments += 1
            self.successful_deployments += 1

            self.logger.info(
                f"Pipeline complete for {module_name}: "
                f"{artifact.size_bytes} bytes, {duration_seconds:.2f}s"
            )

            return artifact

        except Exception as e:
            self.logger.error(f"Pipeline failed for {python_file}: {e}")

            self.total_deployments += 1
            self.failed_deployments += 1

            await self.publish_metrics(
                "portalis_wasm_pipeline_duration_seconds",
                (datetime.now() - start_time).total_seconds(),
                labels={"module": module_name, "status": "failed"}
            )

            raise

    async def batch_process(
        self,
        python_files: List[Path],
        max_concurrent: int = None
    ) -> List[WasmArtifact]:
        """
        Process multiple Python files in parallel

        Args:
            python_files: List of Python source files
            max_concurrent: Maximum concurrent pipelines

        Returns:
            List of WASM artifacts
        """
        max_concurrent = max_concurrent or self.config.max_concurrent_deployments

        self.logger.info(f"Batch processing {len(python_files)} files (max {max_concurrent} concurrent)")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(python_file: Path):
            async with semaphore:
                return await self.process_pipeline(python_file)

        tasks = [process_with_semaphore(f) for f in python_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        artifacts = [r for r in results if isinstance(r, WasmArtifact)]
        errors = [r for r in results if isinstance(r, Exception)]

        self.logger.info(
            f"Batch processing complete: {len(artifacts)} successful, {len(errors)} failed"
        )

        return artifacts

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        success_rate = (
            self.successful_deployments / self.total_deployments
            if self.total_deployments > 0 else 0.0
        )

        return {
            "total_deployments": self.total_deployments,
            "successful_deployments": self.successful_deployments,
            "failed_deployments": self.failed_deployments,
            "success_rate": success_rate
        }


# CLI interface
async def main():
    import argparse

    parser = argparse.ArgumentParser(description="WASM Pipeline Automation")
    parser.add_argument("python_files", nargs="+", type=Path, help="Python source files")
    parser.add_argument("--module-name", help="Module name (for single file)")
    parser.add_argument("--no-deploy", action="store_true", help="Skip Omniverse deployment")
    parser.add_argument("--concurrent", type=int, default=10, help="Max concurrent pipelines")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize pipeline
    config = PipelineConfig(auto_deploy=not args.no_deploy)
    pipeline = WasmPipelineAutomation(config)

    # Process files
    if len(args.python_files) == 1 and args.module_name:
        artifact = await pipeline.process_pipeline(
            args.python_files[0],
            module_name=args.module_name
        )
        print(f"✓ Artifact created: {artifact.artifact_id}")
        print(f"  WASM: {artifact.wasm_binary_path}")
        print(f"  Size: {artifact.size_bytes} bytes")
    else:
        artifacts = await pipeline.batch_process(
            args.python_files,
            max_concurrent=args.concurrent
        )
        print(f"✓ Batch processing complete: {len(artifacts)} artifacts")

    # Print stats
    stats = pipeline.get_stats()
    print(f"\nPipeline Statistics:")
    print(f"  Total: {stats['total_deployments']}")
    print(f"  Successful: {stats['successful_deployments']}")
    print(f"  Failed: {stats['failed_deployments']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
