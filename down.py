import os
import yt_dlp

DOWNLOAD_DIR = "downloads"

# ============ CONFIGURABLE LIMITS ============
# You can change these values based on your hosting
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB (2048 MB)
MAX_DURATION = 7200  # 120 minutes (2 hours)

# Warning thresholds (not rejections, just warnings)
WARN_FILE_SIZE = 1.5 * 1024 * 1024 * 1024  # 1.5 GB - Show warning
WARN_DURATION = 5400  # 90 minutes - Show warning

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_video_from_url(url, status_callback=None, max_size=None, max_duration=None):
    """
    Download video from URL with configurable size and duration limits
    
    Args:
        url: Video URL to download
        status_callback: Optional async function to update status messages
        max_size: Override MAX_FILE_SIZE (in bytes)
        max_duration: Override MAX_DURATION (in seconds)
    
    Returns:
        str: Path to downloaded video file
    
    Raises:
        Exception: If video exceeds limits or download fails
    """
    
    # Use provided limits or defaults
    size_limit = max_size if max_size is not None else MAX_FILE_SIZE
    duration_limit = max_duration if max_duration is not None else MAX_DURATION
    
    # First, get video info without downloading
    try:
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Check file size
            size = info.get("filesize") or info.get("filesize_approx")
            
            if size:
                size_mb = round(size / (1024 * 1024), 2)
                size_gb = round(size / (1024 * 1024 * 1024), 2)
                
                if size > size_limit:
                    limit_gb = round(size_limit / (1024 * 1024 * 1024), 1)
                    raise Exception(
                        f"❌ Video exceeds {limit_gb} GB limit\n"
                        f"📦 Size: {size_gb} GB ({size_mb} MB)\n"
                        f"⚠️ Please use a smaller video"
                    )
            else:
                size_mb = None
                size_gb = None
            
            # Check duration
            duration = info.get("duration", 0)
            
            if duration > duration_limit:
                minutes = round(duration / 60, 1)
                limit_minutes = round(duration_limit / 60, 1)
                raise Exception(
                    f"❌ Video longer than {limit_minutes} minutes\n"
                    f"⏱️ Duration: {minutes} minutes\n"
                    f"⚠️ Please use a shorter video"
                )
            
            # Check if video has audio/video streams
            if not info.get("formats"):
                raise Exception(
                    "❌ No downloadable formats found\n"
                    "⚠️ The URL may not be a valid video"
                )
                
    except Exception as e:
        if "Video exceeds" in str(e) or "longer than" in str(e):
            raise
        raise Exception(f"❌ Failed to get video info: {str(e)}")

    # Prepare download options with chunked downloading for large files
    ydl_opts = {
        "outtmpl": f"{DOWNLOAD_DIR}/%(id)s.%(ext)s",
        "quiet": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
        "restrictfilenames": True,
        "extractor_args": {
            "generic": {
                "impersonate": ["chrome"]
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/137.0 Safari/537.36"
            )
        },
        # For large files, use chunked downloading
        "fragment_retries": 10,
        "retries": 10,
        "continuedl": True,  # Resume partial downloads
        "progress_hooks": [lambda d: _progress_hook(d, status_callback)]
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            # Get the downloaded file path
            video_path = ydl.prepare_filename(info)
            
            # Check if file exists with the expected name
            if os.path.exists(video_path):
                return video_path
            
            # Try different extensions if the file was merged
            base = os.path.splitext(video_path)[0]
            for ext in (".mp4", ".mkv", ".webm", ".mov", ".avi"):
                candidate = base + ext
                if os.path.exists(candidate):
                    return candidate
            
            raise FileNotFoundError("Downloaded file not found")
            
    except Exception as e:
        raise Exception(f"❌ Download failed: {str(e)}")

def _progress_hook(d, status_callback=None):
    """
    Progress hook for yt-dlp to track download progress
    """
    if d["status"] == "downloading":
        try:
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                downloaded = d.get("downloaded_bytes", 0)
                percent = round((downloaded / total) * 100, 1)
                
                # Calculate remaining time
                speed = d.get("speed", 0)
                if speed and speed > 0:
                    remaining = (total - downloaded) / speed
                    if remaining < 60:
                        time_text = f"{int(remaining)}s"
                    elif remaining < 3600:
                        time_text = f"{int(remaining/60)}m {int(remaining%60)}s"
                    else:
                        time_text = f"{int(remaining/3600)}h {int((remaining%3600)/60)}m"
                else:
                    time_text = "Unknown"
                
                # Log progress (for debugging)
                # print(f"Download: {percent}% - {time_text} remaining")
                
        except:
            pass

def get_video_info(url, max_size=None, max_duration=None):
    """
    Get video information without downloading
    
    Args:
        url: Video URL
        max_size: Override MAX_FILE_SIZE
        max_duration: Override MAX_DURATION
    
    Returns:
        dict: Video information including size, duration, title, etc.
    """
    try:
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            
            size = info.get("filesize") or info.get("filesize_approx")
            duration = info.get("duration", 0)
            
            # Use provided limits or defaults
            size_limit = max_size if max_size is not None else MAX_FILE_SIZE
            duration_limit = max_duration if max_duration is not None else MAX_DURATION
            
            return {
                "title": info.get("title", "Unknown"),
                "duration": duration,
                "duration_minutes": round(duration / 60, 1),
                "duration_hours": round(duration / 3600, 2),
                "size": size,
                "size_mb": round(size / (1024 * 1024), 2) if size else None,
                "size_gb": round(size / (1024 * 1024 * 1024), 2) if size else None,
                "is_valid": (
                    (size is None or size <= size_limit) and 
                    duration <= duration_limit
                ),
                "exceeds_size": size is not None and size > size_limit,
                "exceeds_duration": duration > duration_limit,
                "is_warning": (
                    (size is not None and size > WARN_FILE_SIZE) or
                    duration > WARN_DURATION
                ),
                "formats": len(info.get("formats", [])),
                "thumbnail": info.get("thumbnail"),
                "description": info.get("description", "")[:200],
                "uploader": info.get("uploader", "Unknown"),
                "view_count": info.get("view_count", 0),
                "like_count": info.get("like_count", 0),
                "upload_date": info.get("upload_date", "Unknown"),
            }
    except Exception as e:
        return {
            "error": str(e),
            "is_valid": False
        }

async def download_with_status(url, status_msg, max_size=None, max_duration=None):
    """
    Download video with status updates
    
    Args:
        url: Video URL
        status_msg: Message object to update with status
        max_size: Override MAX_FILE_SIZE
        max_duration: Override MAX_DURATION
    
    Returns:
        str: Path to downloaded video
    """
    # Get video info first
    info = get_video_info(url, max_size, max_duration)
    
    if info.get("error"):
        await status_msg.edit(f"❌ Error getting video info: {info['error']}")
        return None
    
    if not info.get("is_valid"):
        error_msg = "❌ **Video rejected:**\n"
        if info.get("exceeds_size"):
            limit_gb = round((max_size or MAX_FILE_SIZE) / (1024 * 1024 * 1024), 1)
            error_msg += f"• Size: **{info['size_gb']} GB** (max {limit_gb} GB)\n"
        if info.get("exceeds_duration"):
            limit_min = round((max_duration or MAX_DURATION) / 60, 1)
            error_msg += f"• Duration: **{info['duration_minutes']} min** (max {limit_min} min)\n"
        error_msg += "\n⚠️ Please use a smaller/shorter video"
        await status_msg.edit(error_msg)
        return None
    
    # Build info message
    size_text = f"{info['size_mb']} MB" if info['size_mb'] else "Unknown"
    if info['size_gb'] and info['size_gb'] > 1:
        size_text = f"{info['size_gb']} GB ({size_text})"
    
    # Add risk warning for large files
    warning_text = ""
    if info.get("is_warning"):
        warning_text = "\n\n⚠️ **Warning:** Large file may take time and use significant resources"
        if info['size_gb'] and info['size_gb'] > 1.8:
            warning_text = "\n\n⚠️ **High Risk:** Very large file may fail on free hosting"
        elif info['size_gb'] and info['size_gb'] > 1.2:
            warning_text = "\n\n⚠️ **Risk:** Large file may be slow or timeout"
    
    # Show video info to user
    await status_msg.edit(
        f"📹 **Video Info:**\n"
        f"📝 Title: {info['title'][:50]}...\n"
        f"👤 Uploader: {info['uploader']}\n"
        f"📦 Size: {size_text}\n"
        f"⏱️ Duration: {info['duration_minutes']} minutes"
        f"{warning_text}\n\n"
        f"⬇️ Starting download..."
    )
    
    try:
        # Download the video with limits
        video_path = download_video_from_url(url, None, max_size, max_duration)
        
        # Verify download was successful
        if not video_path or not os.path.exists(video_path):
            await status_msg.edit("❌ Download failed - File not found")
            return None
        
        # Check actual file size
        actual_size = os.path.getsize(video_path)
        if actual_size > (max_size or MAX_FILE_SIZE):
            os.remove(video_path)
            await status_msg.edit("❌ Downloaded file exceeds size limit")
            return None
        
        return video_path
        
    except Exception as e:
        await status_msg.edit(f"❌ {str(e)}")
        return None

def set_limits(max_size_gb=None, max_duration_min=None):
    """
    Dynamically change the limits
    
    Args:
        max_size_gb: Max file size in GB (e.g., 2.0 for 2GB)
        max_duration_min: Max duration in minutes (e.g., 120 for 2 hours)
    """
    global MAX_FILE_SIZE, MAX_DURATION
    
    if max_size_gb is not None:
        MAX_FILE_SIZE = int(max_size_gb * 1024 * 1024 * 1024)
    
    if max_duration_min is not None:
        MAX_DURATION = int(max_duration_min * 60)
    
    return MAX_FILE_SIZE, MAX_DURATION

def get_current_limits():
    """
    Get current limits
    """
    return {
        "max_size_bytes": MAX_FILE_SIZE,
        "max_size_mb": round(MAX_FILE_SIZE / (1024 * 1024), 1),
        "max_size_gb": round(MAX_FILE_SIZE / (1024 * 1024 * 1024), 1),
        "max_duration_seconds": MAX_DURATION,
        "max_duration_minutes": round(MAX_DURATION / 60, 1),
        "max_duration_hours": round(MAX_DURATION / 3600, 1),
    }
