#!/bin/bash
gunicorn --bind 0.0.0.0:5000 wsgi:app --daemon
nohup python3 fetch_scheduler.py &