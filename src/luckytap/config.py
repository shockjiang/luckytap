"""Constants and default configuration."""

# Main loop poll interval in seconds
POLL_INTERVAL = 0.05

# OpenCV template matching threshold (0.0–1.0)
MATCH_THRESHOLD = 0.75

# Minimum distance (pixels) between two detections to consider them distinct
DEDUP_DISTANCE = 30

# Delay after clicking an envelope before looking for the "open" button
OPEN_BUTTON_DELAY = 0.3

# Delay after clicking the "open" button before pressing Escape
DISMISS_DELAY = 0.3

# How often to re-query the WeChat window position (seconds)
WINDOW_REQUERY_INTERVAL = 5.0

# Chat area: capture the right portion of the WeChat window.
# WeChat's left sidebar is roughly 25-30% of the window width.
# Use a conservative value to avoid clipping left-aligned messages.
CHAT_AREA_LEFT_RATIO = 0.25

# Default template directory name
DEFAULT_TEMPLATE_DIR = "templates"

# Conversation scanning settings
NUM_CHATS = 10  # Number of sidebar conversations to scan
CONVERSATION_SWITCH_DELAY = 0.5  # Delay after clicking a conversation (seconds)
SIDEBAR_ROW_HEIGHT = 64  # Height of each conversation row in points
SIDEBAR_TOP_OFFSET = 90  # Offset from window top to first conversation in points

# Red packets with center x beyond this ratio of the chat area width
# are considered sent by the user and ignored (right side = self).
SELF_MSG_X_RATIO = 0.5

# Scrolling settings for scanning chat history
SCROLL_STEPS = 5  # Number of scroll steps per conversation
SCROLL_AMOUNT = 5  # Scroll wheel units per step (positive = scroll up = older messages)
SCROLL_DELAY = 0.3  # Delay after each scroll to let the UI settle
