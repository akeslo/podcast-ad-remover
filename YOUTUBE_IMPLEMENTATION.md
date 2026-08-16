# YouTube Channel Support Implementation

## Overview
Successfully implemented YouTube channel support for the podcast ad remover application. Videos are now downloaded, processed for ad removal using SponsorBlock + AI fallback, and served through both web video player and direct downloads.

## Implementation Summary

### 1. Database Schema ✅
- Added `source_type` column to subscriptions ('rss' or 'youtube')
- Added `video_id`, `is_video`, `thumbnail_url` columns to episodes
- Added `enable_sponsorblock` and `sponsorblock_categories` settings
- Added `listen_log` table for video view tracking

### 2. Models ✅
- Updated `Subscription` model with `source_type` field
- Updated `Episode` model with `video_id`, `is_video`, `thumbnail_url` fields

### 3. Dependencies ✅
- Added `yt-dlp>=2024.0.0` to requirements.txt
- **ACTION REQUIRED**: Run `pip install yt-dlp` in your environment

### 4. New Core Modules ✅
- `app/core/youtube_feed.py`: YouTube channel parsing (mirrors feed.py)
- `app/core/sponsorblock.py`: SponsorBlock API client
- `app/core/video.py`: Video processing utilities (mirrors audio.py)

### 5. Processor Updates ✅
- Auto-detects YouTube URLs in `check_feeds()`
- Downloads videos using yt-dlp
- Hybrid ad detection: SponsorBlock first, AI transcription fallback
- Extracts audio from video for transcription when needed
- Removes ad segments from video files using FFmpeg

### 6. Repository Updates ✅
- Updated `create()` to accept `source_type` parameter
- Updated `create_or_ignore()` to include video fields

### 7. Web Routes ✅
- Auto-detects YouTube URLs in `/add` endpoint
- Parses YouTube channel metadata in `setup_subscription()`
- Created `app/api/video_routes.py` for video serving with view tracking
- Registered video router in `app/main.py`

### 8. UI Updates ✅
- Updated search placeholder: "Search for podcasts or paste YouTube channel URL..."
- Updated episode list to show video play buttons for video episodes
- Updated JavaScript to handle video playback links

### 9. RSS Feed Generation ✅
- Updated enclosure type to `video/mp4` for video episodes
- Updated URL routing to use `/video/` path for videos
- Applied to both per-subscription and unified feeds

## Testing Checklist

### Manual Testing Steps:

1. **Add YouTube Channel:**
   - Go to homepage
   - Enter YouTube channel URL (e.g., `https://www.youtube.com/@channel`)
   - Verify channel metadata is extracted
   - Check that recent videos appear in episodes list

2. **SponsorBlock Test:**
   - Find a video with known SponsorBlock segments
   - Process the episode
   - Verify segments are detected and removed

3. **AI Fallback Test:**
   - Process a video without SponsorBlock data
   - Verify audio is extracted and transcribed
   - Verify AI ad detection runs

4. **Video Playback:**
   - Open processed video in web interface
   - Verify video player loads
   - Test seeking/playback
   - Verify ad segments are removed

5. **Download:**
   - Click download button
   - Verify video file downloads correctly

6. **Retention Policy:**
   - Set retention limit
   - Verify old videos are cleaned up
   - Check storage usage

### Edge Cases to Test:
- YouTube Shorts (vertical videos)
- Very long videos (>2 hours)
- Videos without audio
- Private/deleted videos
- Age-restricted content

## Critical Files Modified

| File | Changes |
|------|---------|
| `app/infra/database.py` | Schema migrations for source_type, video fields, SponsorBlock settings |
| `app/core/models.py` | Added video fields to Subscription and Episode models |
| `app/core/processor.py` | YouTube download logic, SponsorBlock integration, video processing |
| `app/infra/repository.py` | Update create methods to include video fields |
| `app/web/router.py` | Auto-detect YouTube URLs, support YouTube in setup_subscription |
| `app/core/rss_gen.py` | Video enclosure support in RSS feeds |
| `app/main.py` | Register video router |
| `app/web/templates/index.html` | Update form placeholder text |
| `app/web/templates/episodes.html` | Add video player UI |
| `requirements.txt` | Add yt-dlp dependency |

## New Files Created

| File | Purpose |
|------|---------|
| `app/core/youtube_feed.py` | YouTube channel parsing (mirrors feed.py) |
| `app/core/sponsorblock.py` | SponsorBlock API client |
| `app/core/video.py` | Video processing utilities (mirrors audio.py) |
| `app/api/video_routes.py` | Video serving endpoints (mirrors audio_routes.py) |

## Known Limitations

1. **FFmpeg Required**: Video processing requires FFmpeg with libx264 and AAC encoders
2. **Storage**: Videos consume significantly more disk space than audio
3. **Processing Time**: Video processing is slower than audio processing
4. **SponsorBlock Coverage**: Not all videos have SponsorBlock data
5. **YouTube API**: No API key required (uses yt-dlp scraping)

## Next Steps

1. Install yt-dlp: `pip install yt-dlp`
2. Restart the application
3. Test adding a YouTube channel
4. Monitor logs for any errors
5. Verify video playback in browser

## Architecture Notes

The implementation follows existing patterns:
- RSS feeds → `FeedManager` | YouTube channels → `YouTubeFeedManager`
- Audio processing → `AudioProcessor` | Video processing → `VideoProcessor`
- Audio serving → `audio_routes.py` | Video serving → `video_routes.py`

This parallel structure makes the codebase easy to maintain and extend.
