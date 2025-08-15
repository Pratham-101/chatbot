#!/bin/bash

# Setup cron job for daily mutual fund data updates
# This script adds a cron job to run the daily updater at 6 PM every day

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create the cron job command
CRON_JOB="0 18 * * * cd $PROJECT_DIR && $PROJECT_DIR/venv/bin/python $PROJECT_DIR/src/services/data/daily_updater.py --manual >> $PROJECT_DIR/daily_updater.log 2>&1"

# Add the cron job
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "✅ Cron job added successfully!"
echo "📅 Daily updates will run at 6:00 PM every day"
echo "📝 Logs will be saved to: $PROJECT_DIR/daily_updater.log"

# Show current cron jobs
echo ""
echo "📋 Current cron jobs:"
crontab -l 