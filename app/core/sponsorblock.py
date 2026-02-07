import httpx
import logging
from typing import List, Dict, Optional
import json

logger = logging.getLogger(__name__)

class SponsorBlockClient:
    """Client for SponsorBlock API to fetch community-reported sponsor segments."""

    BASE_URL = "https://sponsor.ajay.app/api"

    # Category mapping: SponsorBlock -> Internal labels
    CATEGORY_MAP = {
        'sponsor': 'Ad',
        'selfpromo': 'Promo',
        'interaction': 'Interaction',
        'intro': 'Intro',
        'outro': 'Outro',
        'music_offtopic': 'Music',
    }

    @staticmethod
    async def get_segments(video_id: str, categories: List[str] = None) -> List[Dict]:
        """
        Fetch sponsor segments for a YouTube video.

        Args:
            video_id: YouTube video ID
            categories: List of SponsorBlock categories to fetch

        Returns:
            List of segments with start, end, label fields
        """
        if categories is None:
            categories = ['sponsor', 'selfpromo', 'interaction', 'intro', 'outro']

        try:
            categories_json = json.dumps(categories)
            url = f"{SponsorBlockClient.BASE_URL}/skipSegments"
            params = {
                'videoID': video_id,
                'categories': categories_json
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)

                # 404 means no segments found (not an error)
                if response.status_code == 404:
                    logger.info(f"No SponsorBlock segments found for video {video_id}")
                    return []

                response.raise_for_status()
                data = response.json()

                # Convert to internal format
                segments = []
                for item in data:
                    segment = item.get('segment', [])
                    if len(segment) >= 2:
                        category = item.get('category', 'sponsor')
                        label = SponsorBlockClient.CATEGORY_MAP.get(category, 'Ad')

                        segments.append({
                            'start': float(segment[0]),
                            'end': float(segment[1]),
                            'label': label,
                            'reason': f"SponsorBlock: {category}"
                        })

                logger.info(f"Found {len(segments)} SponsorBlock segments for video {video_id}")
                return segments

        except httpx.HTTPStatusError as e:
            logger.warning(f"SponsorBlock API error for {video_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch SponsorBlock data for {video_id}: {e}")
            return []
