"""One-off script to fully remove a channel from SQLite + ChromaDB."""
from app.db.sqlite import get_engine, delete_channel, get_videos_by_channel
from app.db.vectordb import VectorDB
from sqlmodel import Session

CHANNEL_ID = "UCYO_jab_esuFRV4b17AJtAw"  # 3Blue1Brown

engine = get_engine()
with Session(engine) as session:
    videos = get_videos_by_channel(session, CHANNEL_ID)
    video_ids = [v.youtube_id for v in videos]
    print(f"Found {len(video_ids)} videos to remove")

    # Remove chunks from ChromaDB
    if video_ids:
        vdb = VectorDB()
        vdb.delete_chunks_for_videos(video_ids)
        print(f"Removed chunks from ChromaDB for {len(video_ids)} videos")

    # Remove from SQLite (cascades jobs + videos + channel)
    removed = delete_channel(session, CHANNEL_ID)
    print(f"SQLite deletion: {'success' if removed else 'channel not found'}")

print("Done.")
