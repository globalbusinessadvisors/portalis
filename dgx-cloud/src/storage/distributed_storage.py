"""
Distributed Storage Integration for Portalis DGX Cloud
S3/GCS storage for models, artifacts, and results with distributed caching
"""

import os
import hashlib
import pickle
from typing import Dict, List, Optional, Any, BinaryIO
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import json

import boto3
from botocore.exceptions import ClientError
import redis
from loguru import logger


@dataclass
class StorageConfig:
    """Storage configuration"""
    # S3 configuration
    s3_bucket: str
    s3_region: str = "us-east-1"
    s3_endpoint: Optional[str] = None

    # GCS configuration (alternative)
    gcs_bucket: Optional[str] = None
    gcs_project: Optional[str] = None

    # Cache configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl_seconds: int = 3600  # 1 hour default

    # Prefixes
    models_prefix: str = "models/"
    cache_prefix: str = "cache/"
    results_prefix: str = "results/"
    artifacts_prefix: str = "artifacts/"


class DistributedCache:
    """
    Distributed Redis cache for translation results and embeddings
    Supports multi-level caching with LRU eviction
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize distributed cache

        Args:
            config: Storage configuration
        """
        self.config = config

        # Redis client
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            decode_responses=False  # Binary mode for pickle
        )

        # Redis cluster for high availability (optional)
        try:
            from redis import RedisCluster
            nodes = os.getenv("REDIS_CLUSTER_NODES", "").split(",")
            if nodes and nodes[0]:
                self.redis_cluster = RedisCluster(
                    startup_nodes=[
                        {"host": node.split(":")[0], "port": int(node.split(":")[1])}
                        for node in nodes
                    ],
                    decode_responses=False
                )
                logger.info("Connected to Redis cluster")
            else:
                self.redis_cluster = None
        except ImportError:
            self.redis_cluster = None

        # Cache statistics
        self.hits = 0
        self.misses = 0

        logger.info("DistributedCache initialized")

    def _get_client(self):
        """Get Redis client (cluster if available, else single)"""
        return self.redis_cluster if self.redis_cluster else self.redis_client

    def _make_key(self, namespace: str, key: str) -> str:
        """Create cache key with namespace"""
        return f"portalis:cache:{namespace}:{key}"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            namespace: Cache namespace (e.g., 'translation', 'embedding')
            key: Cache key

        Returns:
            Cached value or None
        """
        cache_key = self._make_key(namespace, key)
        client = self._get_client()

        try:
            value = client.get(cache_key)
            if value:
                self.hits += 1
                logger.debug(f"Cache hit: {namespace}:{key}")
                return pickle.loads(value)
            else:
                self.misses += 1
                logger.debug(f"Cache miss: {namespace}:{key}")
                return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.misses += 1
            return None

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache

        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use default)

        Returns:
            True if successful
        """
        cache_key = self._make_key(namespace, key)
        ttl = ttl or self.config.cache_ttl_seconds
        client = self._get_client()

        try:
            serialized = pickle.dumps(value)
            client.setex(cache_key, ttl, serialized)
            logger.debug(f"Cache set: {namespace}:{key} (ttl={ttl}s)")
            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def delete(self, namespace: str, key: str) -> bool:
        """Delete key from cache"""
        cache_key = self._make_key(namespace, key)
        client = self._get_client()

        try:
            client.delete(cache_key)
            logger.debug(f"Cache delete: {namespace}:{key}")
            return True

        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def clear_namespace(self, namespace: str):
        """Clear all keys in namespace"""
        pattern = self._make_key(namespace, "*")
        client = self._get_client()

        try:
            keys = client.keys(pattern)
            if keys:
                client.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys from namespace {namespace}")

        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        client = self._get_client()
        info = client.info("memory")

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
            "max_memory_mb": info.get("maxmemory", 0) / (1024 * 1024)
        }


class S3StorageManager:
    """
    S3 storage manager for models, results, and artifacts
    Handles uploads, downloads, and lifecycle management
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize S3 storage manager

        Args:
            config: Storage configuration
        """
        self.config = config

        # S3 client
        s3_config = {
            "region_name": config.s3_region
        }
        if config.s3_endpoint:
            s3_config["endpoint_url"] = config.s3_endpoint

        self.s3_client = boto3.client("s3", **s3_config)
        self.s3_resource = boto3.resource("s3", **s3_config)

        # Verify bucket exists
        try:
            self.s3_client.head_bucket(Bucket=config.s3_bucket)
            logger.info(f"Connected to S3 bucket: {config.s3_bucket}")
        except ClientError:
            logger.error(f"S3 bucket not found: {config.s3_bucket}")
            raise

    def upload_model(
        self,
        local_path: str,
        model_name: str,
        version: str = "latest",
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload NeMo model to S3

        Args:
            local_path: Local path to model file
            model_name: Model identifier
            version: Model version
            metadata: Optional metadata tags

        Returns:
            S3 URI
        """
        s3_key = f"{self.config.models_prefix}{model_name}/{version}/model.nemo"

        try:
            # Upload with metadata
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata

            self.s3_client.upload_file(
                local_path,
                self.config.s3_bucket,
                s3_key,
                ExtraArgs=extra_args
            )

            s3_uri = f"s3://{self.config.s3_bucket}/{s3_key}"
            logger.info(f"Uploaded model to {s3_uri}")
            return s3_uri

        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            raise

    def download_model(
        self,
        model_name: str,
        local_path: str,
        version: str = "latest"
    ) -> str:
        """
        Download NeMo model from S3

        Args:
            model_name: Model identifier
            local_path: Local path to save model
            version: Model version

        Returns:
            Local path to downloaded model
        """
        s3_key = f"{self.config.models_prefix}{model_name}/{version}/model.nemo"

        try:
            # Create directory if needed
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            # Download
            self.s3_client.download_file(
                self.config.s3_bucket,
                s3_key,
                local_path
            )

            logger.info(f"Downloaded model from s3://{self.config.s3_bucket}/{s3_key}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def upload_result(
        self,
        job_id: str,
        result_data: Dict[str, Any],
        tenant_id: str = "default"
    ) -> str:
        """
        Upload translation result to S3

        Args:
            job_id: Job identifier
            result_data: Result data to upload
            tenant_id: Tenant identifier

        Returns:
            S3 URI
        """
        s3_key = f"{self.config.results_prefix}{tenant_id}/{job_id}/result.json"

        try:
            # Serialize result
            result_json = json.dumps(result_data, indent=2)

            # Upload
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Body=result_json.encode("utf-8"),
                ContentType="application/json",
                Metadata={
                    "job_id": job_id,
                    "tenant_id": tenant_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            s3_uri = f"s3://{self.config.s3_bucket}/{s3_key}"
            logger.info(f"Uploaded result to {s3_uri}")
            return s3_uri

        except Exception as e:
            logger.error(f"Failed to upload result: {e}")
            raise

    def download_result(self, job_id: str, tenant_id: str = "default") -> Dict[str, Any]:
        """
        Download translation result from S3

        Args:
            job_id: Job identifier
            tenant_id: Tenant identifier

        Returns:
            Result data
        """
        s3_key = f"{self.config.results_prefix}{tenant_id}/{job_id}/result.json"

        try:
            response = self.s3_client.get_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key
            )

            result_json = response["Body"].read().decode("utf-8")
            return json.loads(result_json)

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.error(f"Result not found: {job_id}")
                return None
            raise

    def upload_artifact(
        self,
        local_path: str,
        artifact_name: str,
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload arbitrary artifact (e.g., WASM binary, logs)

        Args:
            local_path: Local path to artifact
            artifact_name: Artifact identifier
            job_id: Optional job ID for organization
            metadata: Optional metadata

        Returns:
            S3 URI
        """
        if job_id:
            s3_key = f"{self.config.artifacts_prefix}{job_id}/{artifact_name}"
        else:
            s3_key = f"{self.config.artifacts_prefix}{artifact_name}"

        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata

            self.s3_client.upload_file(
                local_path,
                self.config.s3_bucket,
                s3_key,
                ExtraArgs=extra_args
            )

            s3_uri = f"s3://{self.config.s3_bucket}/{s3_key}"
            logger.info(f"Uploaded artifact to {s3_uri}")
            return s3_uri

        except Exception as e:
            logger.error(f"Failed to upload artifact: {e}")
            raise

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix=self.config.models_prefix
            )

            models = []
            for obj in response.get("Contents", []):
                # Extract model info from key
                key = obj["Key"]
                parts = key.replace(self.config.models_prefix, "").split("/")

                if len(parts) >= 2:
                    models.append({
                        "name": parts[0],
                        "version": parts[1],
                        "size_mb": obj["Size"] / (1024 * 1024),
                        "last_modified": obj["LastModified"].isoformat(),
                        "s3_key": key
                    })

            return models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def cleanup_old_results(self, days: int = 30):
        """
        Clean up results older than specified days

        Args:
            days: Number of days to keep
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix=self.config.results_prefix
            )

            deleted = 0
            for obj in response.get("Contents", []):
                if obj["LastModified"].replace(tzinfo=None) < cutoff:
                    self.s3_client.delete_object(
                        Bucket=self.config.s3_bucket,
                        Key=obj["Key"]
                    )
                    deleted += 1

            logger.info(f"Cleaned up {deleted} old result objects")

        except Exception as e:
            logger.error(f"Failed to cleanup results: {e}")


class DistributedStorageManager:
    """
    Unified storage manager combining S3 and distributed caching
    Provides high-performance access to models, results, and artifacts
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize distributed storage manager

        Args:
            config: Storage configuration
        """
        self.config = config
        self.cache = DistributedCache(config)
        self.s3 = S3StorageManager(config)

        logger.info("DistributedStorageManager initialized")

    async def get_translation(self, code_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached translation result

        Args:
            code_hash: Hash of source code

        Returns:
            Cached translation or None
        """
        # Try cache first
        cached = self.cache.get("translation", code_hash)
        if cached:
            return cached

        # Not in cache
        return None

    async def store_translation(
        self,
        code_hash: str,
        translation: Dict[str, Any],
        ttl: int = 3600
    ):
        """
        Store translation in cache

        Args:
            code_hash: Hash of source code
            translation: Translation result
            ttl: Time-to-live in seconds
        """
        self.cache.set("translation", code_hash, translation, ttl=ttl)

    async def get_embedding(self, text: str) -> Optional[Any]:
        """Get cached embedding"""
        # Use hash of text as key
        key = hashlib.sha256(text.encode()).hexdigest()
        return self.cache.get("embedding", key)

    async def store_embedding(self, text: str, embedding: Any, ttl: int = 7200):
        """Store embedding in cache"""
        key = hashlib.sha256(text.encode()).hexdigest()
        self.cache.set("embedding", key, embedding, ttl=ttl)

    def compute_code_hash(self, code: str) -> str:
        """
        Compute stable hash of code for caching

        Args:
            code: Source code

        Returns:
            SHA256 hash
        """
        # Normalize whitespace for stable hashing
        normalized = " ".join(code.split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        cache_stats = self.cache.get_stats()
        models = self.s3.list_models()

        return {
            "cache": cache_stats,
            "s3": {
                "bucket": self.config.s3_bucket,
                "models_count": len(models),
                "total_model_size_gb": sum(m["size_mb"] for m in models) / 1024
            }
        }
