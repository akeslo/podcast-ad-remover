from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from app.core.models import Subscription, SubscriptionCreate
from app.infra.repository import SubscriptionRepository, EpisodeRepository
from app.core.feed import FeedManager
from app.core.processor import Processor
from app.core.search import PodcastSearcher
from pydantic import BaseModel
import shutil
import os
from app.core.config import settings
# The same two dependencies the HTML routes use, imported rather than
# re-declared so there is exactly one definition of "guarded" in the app.
# `app.web.router` does not import anything under `app.api`, so this edge is
# one-way and creates no import cycle.
#
# What they actually buy here is what they buy there, and no more: with
# `auth_enabled = 1` `require_auth`/`require_admin` are real boundaries; with
# `auth_enabled = 0` (standalone, the default) `require_auth` is a deliberate
# no-op and the same-origin check is the entire boundary - it stops drive-by
# cross-site posts and blind scripted loops (curl sends no Origin) and nothing
# else. The remaining boundary there is the IP allowlist in `auth_middleware`.
from app.web.router import require_admin_action, require_user_action


router = APIRouter()
repo = SubscriptionRepository()

# Helper to get processor (in a real app, use dependency injection)
def get_processor():
    return Processor()

@router.get("/subscriptions", response_model=List[Subscription])
async def list_subscriptions():
    return repo.get_all()

@router.post("/subscriptions", response_model=Subscription, dependencies=[Depends(require_user_action)])
async def create_subscription(sub: SubscriptionCreate, initial_count: int = 5):
    if not (0 <= initial_count <= 100):
        raise HTTPException(status_code=400, detail="initial_count must be between 0 and 100")

    existing = repo.get_by_url(sub.feed_url)
    if existing:
        raise HTTPException(status_code=400, detail="Subscription already exists")

    try:
        # Parse feed to get title
        title, slug, image_url, description = FeedManager.parse_feed(sub.feed_url)

        # Save to DB
        new_sub = repo.create(sub, title, slug, image_url)
        
        # Trigger initial check
        proc = get_processor()
        await proc.check_feeds(subscription_id=new_sub.id, limit=initial_count)
        
        # Processor loop will pick up 'pending' items automatically
        

        
        return new_sub
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/subscriptions/{id}", dependencies=[Depends(require_admin_action)])
async def delete_subscription(id: int):
    """Delete a subscription and its episodes."""
    sub = repo.get_by_id(id)
    
    if sub:
        # 1. Delete all episodes using Processor (cleans DB + all artifact files)
        proc = get_processor()
        ep_repo = EpisodeRepository()
        
        # We need a way to get all episodes IDs first
        # Assuming get_by_subscription returns models with IDs
        episodes = ep_repo.get_by_subscription(sub.id)
        for ep in episodes:
            await proc.delete_episode(ep.id)
            
        # 2. Delete subscription-level folders/files
        
        # Audio directory (Subscription folder)
        dir_path = os.path.join(settings.AUDIO_DIR, sub.slug)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                print(f"Error deleting directory {dir_path}: {e}")
        
        # Feed file
        feed_path = os.path.join(settings.FEEDS_DIR, f"{sub.slug}.xml")
        if os.path.exists(feed_path):
            try:
                os.remove(feed_path)
            except Exception as e:
                 print(f"Error deleting feed file {feed_path}: {e}")

        # 3. Delete Subscription from DB
        repo.delete(id)
    
    return {"status": "deleted"}

@router.delete("/episodes/{id}", dependencies=[Depends(require_user_action)])
async def delete_episode(id: int):
    """Delete a specific episode and its files."""
    proc = get_processor()
    success = await proc.delete_episode(id)
    if not success:
        raise HTTPException(status_code=404, detail="Episode not found")
    return {"status": "deleted"}

@router.post("/subscriptions/{id}/check", dependencies=[Depends(require_user_action)])
async def check_subscription_updates(id: int, background_tasks: BackgroundTasks):
    """Trigger a check for new episodes."""
    # We allow running check_feeds in background task because it's I/O bound (network)
    # and doesn't use heavy CPU. The background loop will then pick up any new pending episodes.
    proc = get_processor()
    background_tasks.add_task(proc.check_feeds, subscription_id=id)
    return {"status": "check_triggered"}

@router.post("/episodes/{id}/process", dependencies=[Depends(require_user_action)])
async def process_episode(id: int, skip_transcription: bool = False):
    """Manually trigger processing for an episode."""
    ep_repo = EpisodeRepository()
    
    import json
    flags = {'skip_transcription': skip_transcription}
    flags_json = json.dumps(flags)
    
    # Using reset_status to ensure clean state but with flags
    ep_repo.reset_status(id, processing_flags=flags_json)
    ep_repo.update_status(id, "pending")
    
    # Background processor (polling) will pick this up
    return {"status": "processing_triggered"}

@router.post("/episodes/{id}/cancel", dependencies=[Depends(require_user_action)])
async def cancel_episode(id: int):
    """Cancel processing and reset status (Soft Delete)."""
    proc = get_processor()
    success = await proc.delete_episode(id)
    if not success:
        raise HTTPException(status_code=404, detail="Episode not found")
    return {"status": "cancelled"}

class SearchQuery(BaseModel):
    query: str

@router.post("/search", dependencies=[Depends(require_user_action)])
async def search_podcasts(q: SearchQuery):
    return await PodcastSearcher.search(q.query)

@router.post("/episodes/{id}/track-listen", dependencies=[Depends(require_user_action)])
async def track_listen(id: int):
    """Increment listen count for an episode."""
    ep_repo = EpisodeRepository()
    ep = ep_repo.get_by_id(id)
    if not ep:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    ep_repo.increment_listen_count(id)
    return {"status": "tracked", "episode_id": id}
