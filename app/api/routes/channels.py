from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.api.models import AddChannelRequest, AddChannelResponse, ChannelResponse
from app.db.sqlite import get_session, list_channels
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter()

@router.post("/channels", response_model=AddChannelResponse)
def add_channel(
  request: AddChannelRequest,
  session: Session = Depends(get_session),
):
  try:
    pipeline = IngestionPipeline()
    result = pipeline.add_channel(
      channel_input=request.channel_input,
      max_videos=request.max_videos
    )

    return AddChannelResponse(
      **result,
      message=f"Successfully ingested {result['videos_ingested']}"
    )
  
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@router.get("/channels", response_model=list[ChannelResponse])
def get_channels(session: Session = Depends(get_session)):
  channels = list_channels(session)

  return [
    ChannelResponse(
      youtube_id=ch.youtube_id,
      name=ch.name,
      url=ch.url,
      last_checked_at=ch.last_checked_at.isoformat() if ch.last_checked_at else None,
      created_at=ch.created_at.isoformat(),
    )
    for ch in channels
  ]