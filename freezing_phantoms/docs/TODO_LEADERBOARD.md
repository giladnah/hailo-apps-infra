# TODO: Leaderboard System

## Overview

Implement a web-based leaderboard system hosted locally on the game device to track and display the best players. The leaderboard will show:
- Player names/IDs
- Best scores (game completion times)
- Player images captured during gameplay
- Win/loss statistics
- Overall ranking

## Status

**Status**: Not Started
**Priority**: Low (Future Enhancement)
**Target Version**: v2.0+

## Requirements

### 1. Player Image Capture
- Capture player image during gameplay
- Capture at game start (intro/setup phase)
- Store image with player ID/tracking ID
- Format: JPEG, 640x640px (matching camera resolution)
- Storage: `data/player_images/{player_id}_{timestamp}.jpg`

### 2. Score Tracking
- Track player performance metrics:
  - Game completion time
  - Games won
  - Games played
  - Best time
  - Average time
- Store in JSON/CSV format
- Storage: `data/scores.json`

### 3. Database/Storage Structure
```json
{
  "players": [
    {
      "id": "tracking_123",
      "name": "Player 1",
      "image_path": "data/player_images/tracking_123_20241201_143022.jpg",
      "best_time": 45.3,
      "games_won": 5,
      "games_played": 12,
      "average_time": 52.1,
      "last_played": "2024-12-01T14:30:22Z"
    }
  ]
}
```

### 4. Web Interface
- Simple, responsive web UI
- Display top 10 players
- Show player images in leaderboard
- Auto-refresh every 5-10 seconds
- Accessible at: `http://localhost:8080/leaderboard`

### 5. Web Server
- Lightweight HTTP server
- Serve static files (HTML, CSS, JS)
- API endpoint for leaderboard data: `/api/leaderboard`
- API endpoint for player images: `/api/images/{filename}`
- Host on local network (access from other devices)

## Implementation Plan

### Phase 1: Data Collection
- [ ] Modify `game_state.py` to capture player images at start
- [ ] Add image capture logic to `data_extractor.py` or create new module
- [ ] Implement score tracking in `game_state.py`
- [ ] Store data in persistent format (JSON)

### Phase 2: Web Backend
- [ ] Create `web/` directory for web application
- [ ] Implement Flask/FastAPI server (lightweight)
- [ ] Create API endpoints:
  - `GET /api/leaderboard` - Return sorted player data
  - `GET /api/images/<filename>` - Serve player images
  - `GET /api/stats` - Return game statistics
- [ ] Handle CORS for cross-origin requests

### Phase 3: Web Frontend
- [ ] Create `templates/leaderboard.html` - Main leaderboard page
- [ ] Style with CSS (modern, mobile-friendly)
- [ ] JavaScript for auto-refresh and data fetching
- [ ] Display player cards with:
  - Player image
  - Name/ID
  - Best time
  - Win rate
  - Rank

### Phase 4: Integration
- [ ] Add web server startup to `main.py`
- [ ] Optional: Add command-line flag `--enable-leaderboard`
- [ ] Test on target device
- [ ] Document in README

## Technical Specifications

### Web Server Options
1. **Flask** (Recommended - Simple)
   - Easy to set up
   - Good for serving static files
   - Minimal dependencies

2. **FastAPI** (Alternative)
   - Modern async support
   - Auto-generated API docs
   - Better for API-heavy applications

3. **Built-in HTTP Server** (Python 3+)
   - No dependencies
   - Limited functionality
   - Suitable for static files only

### Directory Structure
```
freezing_phantoms/
├── web/
│   ├── __init__.py
│   ├── server.py          # Web server application
│   ├── templates/
│   │   └── leaderboard.html
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── app.js
│   └── api.py            # API endpoints
├── data/
│   ├── player_images/    # Player photos
│   ├── scores.json       # Player data
│   └── stats.json        # Game statistics
└── leaderboard.py        # Leaderboard logic (if needed)
```

## UI Mockup

```
╔═══════════════════════════════════════════════════╗
║     🏆 Freezing Phantoms - Leaderboard 🏆         ║
╠═══════════════════════════════════════════════════╣
║                                                    ║
║  #1  [👤 Image] Player 123     32.5s  ████████░░  ║
║      Best Time: 32.5s  Wins: 8/15                  ║
║                                                    ║
║  #2  [👤 Image] Player 456     35.2s  ███████░░░  ║
║      Best Time: 35.2s  Wins: 6/12                  ║
║                                                    ║
║  #3  [👤 Image] Player 789     38.1s  ██████░░░░  ║
║      Best Time: 38.1s  Wins: 4/10                  ║
║                                                    ║
║  ...                                               ║
║                                                    ║
║  Auto-refresh: On  ●                               ║
╚═══════════════════════════════════════════════════╝
```

## Future Enhancements
- QR code display for easy mobile access
- Export leaderboard data
- Player badges/achievements
- Historical statistics/graphs
- Multi-day tournament mode
- Player registration (name input)
- Share results to social media

## Considerations
- **Privacy**: Player images may contain children - ensure compliance with privacy regulations
- **Performance**: Server should have minimal impact on game performance
- **Storage**: Implement cleanup for old images/scores (retain top N players)
- **Security**: No external access by default (localhost only)
- **Dependencies**: Minimize additional packages for easy deployment

## Testing Checklist
- [ ] Images captured correctly during gameplay
- [ ] Score data persisted correctly
- [ ] Web server starts without errors
- [ ] Leaderboard displays accurate data
- [ ] Images load correctly in browser
- [ ] Auto-refresh works
- [ ] Mobile-responsive design
- [ ] No performance impact on game
- [ ] Works on target device (Raspberry Pi)

## Documentation Updates Needed
- Update README.md with leaderboard section
- Add web server setup instructions
- Document data directory structure
- Add API documentation (if using FastAPI)
