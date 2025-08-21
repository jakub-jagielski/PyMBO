#!/usr/bin/env python3
"""
Simple launcher script for PyMBO
Usage: python launch_pymbo.py
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Launch PyMBO
from pymbo.launcher import main

if __name__ == "__main__":
    print("🚀 Launching PyMBO v3.6.6...")
    print("=" * 50)
    exit(main())