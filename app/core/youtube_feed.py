import logging
from typing import Optional, Tuple, List, Dict
from datetime import datetime
import yt_dlp

logger = logging.getLogger(__name__)

def slugify(text: str) -> str:
    """Convert text to a filename-friendly slug."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text).strip('-')
    return text

class YouTubeFeedManager:
    @staticmethod
    def parse_channel(channel_url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse YouTube channel and return (title, slug, image_url, description)."""
        ydl_opts = {
            'quiet': True,
            'extract_flat': 'in_playlist',
            'playlistend': 1,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)

                if not info:
                    raise ValueError("Could not extract channel information")

                title = info.get('title') or info.get('channel') or 'Unknown Channel'
                # Prefix slug with 'yt-' to avoid conflicts with podcast names
                slug = f"yt-{slugify(title)}"
                description = info.get('description', '')

                # Get thumbnail (prefer channel thumbnail)
                thumbnails = info.get('thumbnails', [])
                image_url = thumbnails[-1]['url'] if thumbnails else None

                # Prefix title with 'YT: ' for visual distinction
                display_title = f"YT: {title}"

                return display_title, slug, image_url, description

        except Exception as e:
            logger.error(f"Error parsing YouTube channel: {e}")
            raise ValueError(f"Invalid YouTube channel: {str(e)}")

    @staticmethod
    def parse_videos(channel_url: str, limit: int = 5) -> List[Dict]:
        """Parse recent videos from YouTube channel."""
        ydl_opts = {
            'quiet': True,
            'extract_flat': 'in_playlist',
            'playlistend': limit,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)

                if not info or 'entries' not in info:
                    return []

                videos = []
                for entry in info['entries'][:limit]:
                    if not entry:
                        continue

                    # Parse upload date
                    pub_date = None
                    if entry.get('upload_date'):
                        try:
                            pub_date = datetime.strptime(entry['upload_date'], '%Y%m%d')
                        except:
                            pass

                    video_data = {
                        'guid': entry['id'],
                        'title': entry.get('title', 'Unknown Video'),
                        'pub_date': pub_date,
                        'original_url': f"https://www.youtube.com/watch?v={entry['id']}",
                        'duration': int(entry.get('duration', 0)),
                        'description': entry.get('description', ''),
                        'file_size': 0,
                        'video_id': entry['id'],
                        'thumbnail_url': entry.get('thumbnail'),
                    }
                    videos.append(video_data)

                return videos

        except Exception as e:
            logger.error(f"Error parsing YouTube videos: {e}")
            return []
