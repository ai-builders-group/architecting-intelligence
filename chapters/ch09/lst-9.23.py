# --8<-- [start:lst-9.23]
# path/to/replay_controller.py
from typing import Iterator, Protocol
from datetime import datetime, UTC

class ReplayFailed(RuntimeError):
    """A failed event stops the replay; the run resumes from this offset."""
    def __init__(self, offset: int):
        super().__init__(f"replay stopped at offset {offset}")
        self.offset = offset

# KeyDestroyed is raised by process_event (lst-9.22) when a document's key or snapshot is gone

class LogRecord(Protocol):
    offset: int  # the offset the log stored for this record
    key: str  # tenant_id:document_id
    event: IngestionEvent

class EventLogReader(Protocol):
    """Abstract interface for event log storage (Kafka, Kinesis, Postgres)."""
    def read_from_offset(self, offset: int, batch_size: int = 100) -> Iterator[LogRecord]: ...

class ReplayController:
    """Orchestrates re-indexing by replaying the event log."""
    
    def __init__(self, event_log: EventLogReader, processor: KappaStreamProcessor):
        self.event_log = event_log
        self.processor = processor
        self.parked: list[tuple[int, str]] = []
    
    def park(self, record: LogRecord, marker: str) -> None:
        """Set a record aside with a marker; the replay does not retry it."""
        self.parked.append((record.offset, marker))
    
    def replay(
        self, 
        start_offset: int = 0, 
        end_offset: int | None = None,
        batch_size: int = 100
    ) -> dict[str, int]:
        """Replay events from the log to rebuild the index.
        
        Returns:
            Statistics dict with processed and erased counts. A failed event
            raises ReplayFailed; the replay never skips it.
        """
        stats = {"processed": 0, "erased": 0}
        
        for record in self.event_log.read_from_offset(start_offset, batch_size):
            event = record.event
            if end_offset is not None and record.offset > end_offset:
                break
            
            if record.event is None:  # compaction tombstone
                self.processor.delete_key(record.key)
                continue
            
            try:
                self.processor.process_event(event)
                stats["processed"] += 1
            except KeyDestroyed:  # crypto-shredded: nothing left to index
                self.park(record, marker="erased")
                stats["erased"] += 1
            except Exception as exc:  # never skip a failed event
                raise ReplayFailed(offset=record.offset) from exc
        
        return stats# --8<-- [end:lst-9.23]
