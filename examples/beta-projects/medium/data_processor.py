"""
Data Processing Library - Medium Complexity Beta Project
Lines of Code: ~500
Complexity: Medium
Purpose: Test intermediate Python features and data processing patterns
"""

from typing import List, Dict, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import json
import statistics
from datetime import datetime

T = TypeVar('T')


class DataFormat(Enum):
    """Supported data formats."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"


@dataclass
class DataRecord:
    """Represents a single data record."""
    id: int
    name: str
    value: float
    timestamp: str
    tags: List[str]

    def to_dict(self) -> Dict[str, any]:
        """Convert record to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> 'DataRecord':
        """Create record from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            value=data['value'],
            timestamp=data['timestamp'],
            tags=data.get('tags', [])
        )


class DataFilter(Generic[T]):
    """Generic filter for data processing."""

    def __init__(self, predicate: Callable[[T], bool]):
        self.predicate = predicate

    def apply(self, items: List[T]) -> List[T]:
        """Apply filter to list of items."""
        return [item for item in items if self.predicate(item)]


class DataProcessor:
    """Main data processing engine."""

    def __init__(self):
        self.records: List[DataRecord] = []
        self.filters: List[DataFilter] = []

    def add_record(self, record: DataRecord) -> None:
        """Add a record to the dataset."""
        self.records.append(record)

    def add_records(self, records: List[DataRecord]) -> None:
        """Add multiple records."""
        self.records.extend(records)

    def get_record_by_id(self, record_id: int) -> Optional[DataRecord]:
        """Find record by ID."""
        for record in self.records:
            if record.id == record_id:
                return record
        return None

    def filter_by_name(self, name: str) -> List[DataRecord]:
        """Filter records by name."""
        return [r for r in self.records if r.name == name]

    def filter_by_value_range(self, min_val: float, max_val: float) -> List[DataRecord]:
        """Filter records by value range."""
        return [r for r in self.records if min_val <= r.value <= max_val]

    def filter_by_tag(self, tag: str) -> List[DataRecord]:
        """Filter records containing specific tag."""
        return [r for r in self.records if tag in r.tags]

    def apply_custom_filter(self, predicate: Callable[[DataRecord], bool]) -> List[DataRecord]:
        """Apply custom filter function."""
        return [r for r in self.records if predicate(r)]

    def calculate_statistics(self) -> Dict[str, float]:
        """Calculate statistics on record values."""
        if not self.records:
            return {
                'count': 0,
                'sum': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0,
                'min': 0.0,
                'max': 0.0
            }

        values = [r.value for r in self.records]

        return {
            'count': len(values),
            'sum': sum(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values)
        }

    def group_by_name(self) -> Dict[str, List[DataRecord]]:
        """Group records by name."""
        groups: Dict[str, List[DataRecord]] = {}
        for record in self.records:
            if record.name not in groups:
                groups[record.name] = []
            groups[record.name].append(record)
        return groups

    def aggregate_by_name(self) -> Dict[str, float]:
        """Aggregate values by name."""
        groups = self.group_by_name()
        return {
            name: sum(r.value for r in records)
            for name, records in groups.items()
        }

    def sort_by_value(self, descending: bool = False) -> List[DataRecord]:
        """Sort records by value."""
        return sorted(self.records, key=lambda r: r.value, reverse=descending)

    def sort_by_timestamp(self, descending: bool = False) -> List[DataRecord]:
        """Sort records by timestamp."""
        return sorted(self.records, key=lambda r: r.timestamp, reverse=descending)

    def to_json(self) -> str:
        """Export records to JSON."""
        data = [r.to_dict() for r in self.records]
        return json.dumps(data, indent=2)

    def from_json(self, json_str: str) -> None:
        """Import records from JSON."""
        data = json.loads(json_str)
        self.records = [DataRecord.from_dict(item) for item in data]

    def clear(self) -> None:
        """Clear all records."""
        self.records.clear()


class DataValidator:
    """Validates data records."""

    @staticmethod
    def validate_record(record: DataRecord) -> List[str]:
        """Validate a record and return list of errors."""
        errors = []

        if record.id < 0:
            errors.append("ID must be non-negative")

        if not record.name or len(record.name.strip()) == 0:
            errors.append("Name cannot be empty")

        if record.value < 0:
            errors.append("Value must be non-negative")

        if not record.timestamp:
            errors.append("Timestamp cannot be empty")

        return errors

    @staticmethod
    def validate_batch(records: List[DataRecord]) -> Dict[int, List[str]]:
        """Validate multiple records."""
        validation_results = {}
        for record in records:
            errors = DataValidator.validate_record(record)
            if errors:
                validation_results[record.id] = errors
        return validation_results


class DataTransformer:
    """Transforms data records."""

    @staticmethod
    def normalize_values(records: List[DataRecord]) -> List[DataRecord]:
        """Normalize record values to 0-1 range."""
        if not records:
            return []

        values = [r.value for r in records]
        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val

        if value_range == 0:
            return records

        normalized = []
        for record in records:
            normalized_value = (record.value - min_val) / value_range
            normalized_record = DataRecord(
                id=record.id,
                name=record.name,
                value=normalized_value,
                timestamp=record.timestamp,
                tags=record.tags.copy()
            )
            normalized.append(normalized_record)

        return normalized

    @staticmethod
    def scale_values(records: List[DataRecord], factor: float) -> List[DataRecord]:
        """Scale all values by a factor."""
        scaled = []
        for record in records:
            scaled_record = DataRecord(
                id=record.id,
                name=record.name,
                value=record.value * factor,
                timestamp=record.timestamp,
                tags=record.tags.copy()
            )
            scaled.append(scaled_record)
        return scaled

    @staticmethod
    def add_tag(records: List[DataRecord], tag: str) -> List[DataRecord]:
        """Add a tag to all records."""
        for record in records:
            if tag not in record.tags:
                record.tags.append(tag)
        return records


class DataAnalyzer:
    """Advanced data analysis."""

    def __init__(self, processor: DataProcessor):
        self.processor = processor

    def find_outliers(self, std_threshold: float = 2.0) -> List[DataRecord]:
        """Find outliers using standard deviation."""
        stats = self.processor.calculate_statistics()
        mean = stats['mean']
        std_dev = stats['std_dev']

        if std_dev == 0:
            return []

        outliers = []
        for record in self.processor.records:
            z_score = abs((record.value - mean) / std_dev)
            if z_score > std_threshold:
                outliers.append(record)

        return outliers

    def get_top_n(self, n: int) -> List[DataRecord]:
        """Get top N records by value."""
        sorted_records = self.processor.sort_by_value(descending=True)
        return sorted_records[:n]

    def get_bottom_n(self, n: int) -> List[DataRecord]:
        """Get bottom N records by value."""
        sorted_records = self.processor.sort_by_value(descending=False)
        return sorted_records[:n]

    def calculate_percentiles(self) -> Dict[str, float]:
        """Calculate percentile values."""
        if not self.processor.records:
            return {}

        values = sorted([r.value for r in self.processor.records])
        n = len(values)

        return {
            'p25': values[n // 4],
            'p50': values[n // 2],
            'p75': values[(3 * n) // 4],
            'p90': values[(9 * n) // 10],
            'p95': values[(19 * n) // 20],
            'p99': values[(99 * n) // 100] if n >= 100 else values[-1]
        }


def create_sample_data() -> List[DataRecord]:
    """Create sample data for testing."""
    return [
        DataRecord(1, "sensor_a", 23.5, "2024-01-01T10:00:00", ["temperature", "indoor"]),
        DataRecord(2, "sensor_a", 24.1, "2024-01-01T11:00:00", ["temperature", "indoor"]),
        DataRecord(3, "sensor_b", 18.3, "2024-01-01T10:00:00", ["temperature", "outdoor"]),
        DataRecord(4, "sensor_b", 19.2, "2024-01-01T11:00:00", ["temperature", "outdoor"]),
        DataRecord(5, "sensor_c", 65.0, "2024-01-01T10:00:00", ["humidity", "indoor"]),
        DataRecord(6, "sensor_c", 63.5, "2024-01-01T11:00:00", ["humidity", "indoor"]),
        DataRecord(7, "sensor_d", 75.2, "2024-01-01T10:00:00", ["humidity", "outdoor"]),
        DataRecord(8, "sensor_d", 72.8, "2024-01-01T11:00:00", ["humidity", "outdoor"]),
    ]


def main():
    """Demonstrate data processing capabilities."""
    # Create processor
    processor = DataProcessor()

    # Load sample data
    sample_data = create_sample_data()
    processor.add_records(sample_data)

    # Statistics
    print("=== Statistics ===")
    stats = processor.calculate_statistics()
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")

    # Filtering
    print("\n=== Filtering ===")
    temp_records = processor.filter_by_tag("temperature")
    print(f"Temperature records: {len(temp_records)}")

    high_value = processor.filter_by_value_range(50, 100)
    print(f"High value records (50-100): {len(high_value)}")

    # Grouping
    print("\n=== Grouping ===")
    by_name = processor.group_by_name()
    for name, records in by_name.items():
        print(f"{name}: {len(records)} records")

    # Aggregation
    print("\n=== Aggregation ===")
    agg = processor.aggregate_by_name()
    for name, total in agg.items():
        print(f"{name}: {total:.2f}")

    # Validation
    print("\n=== Validation ===")
    errors = DataValidator.validate_batch(sample_data)
    if errors:
        print(f"Found {len(errors)} invalid records")
    else:
        print("All records valid")

    # Transformation
    print("\n=== Transformation ===")
    normalized = DataTransformer.normalize_values(sample_data)
    print(f"Normalized {len(normalized)} records")

    # Analysis
    print("\n=== Analysis ===")
    analyzer = DataAnalyzer(processor)
    outliers = analyzer.find_outliers(std_threshold=1.5)
    print(f"Found {len(outliers)} outliers")

    top_3 = analyzer.get_top_n(3)
    print(f"\nTop 3 records:")
    for record in top_3:
        print(f"  {record.name}: {record.value}")

    percentiles = analyzer.calculate_percentiles()
    print(f"\nPercentiles:")
    for p, value in percentiles.items():
        print(f"  {p}: {value:.2f}")

    # Export
    print("\n=== Export ===")
    json_output = processor.to_json()
    print(f"Exported {len(sample_data)} records to JSON ({len(json_output)} bytes)")


if __name__ == "__main__":
    main()
