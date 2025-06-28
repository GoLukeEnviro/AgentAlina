#!/usr/bin/env python3
"""
Health check script for AgentAlina
"""

import sys
import os
import requests
import time

def check_health():
    """Check if the application is healthy"""
    try:
        # Try to connect to the main application
        response = requests.get('http://localhost:8000/health', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                print("✓ Application is healthy")
                return True
            else:
                print("✗ Application reports unhealthy status")
                return False
        else:
            print(f"✗ Health check failed with status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to application")
        return False
    except requests.exceptions.Timeout:
        print("✗ Health check timed out")
        return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def main():
    """Main health check function"""
    # Wait a bit for the application to start
    time.sleep(2)
    
    # Perform health check
    if check_health():
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == '__main__':
    main()