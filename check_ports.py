#!/usr/bin/env python3
"""
Check port status and provide Codespaces access URLs
"""
import subprocess
import os
import urllib.request
import json

def check_port_5000():
    """Check if API is running on port 5000"""
    print("=" * 60)
    print("🔍 CHECKING API SERVER STATUS")
    print("=" * 60)
    
    # Check if port 5000 is in use
    result = subprocess.run(
        ['lsof', '-i', ':5000'],
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print("\n✅ Port 5000 is ACTIVE")
        print("\nProcesses on port 5000:")
        print(result.stdout)
    else:
        print("\n❌ Port 5000 is NOT in use")
        print("Run: python src/api.py")
        return False
    
    # Test the API
    print("\n" + "=" * 60)
    print("🧪 TESTING API ENDPOINTS")
    print("=" * 60)
    
    try:
        with urllib.request.urlopen('http://localhost:5000/health', timeout=5) as response:
            data = json.loads(response.read().decode())
            print("\n✅ /health endpoint:")
            print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"\n❌ API health check failed: {e}")
        return False
    
    return True

def get_codespace_url():
    """Get the Codespace URL for port 5000"""
    print("\n" + "=" * 60)
    print("🌐 CODESPACE ACCESS INFORMATION")
    print("=" * 60)
    
    codespace_name = os.environ.get('CODESPACE_NAME')
    github_codespaces_port_forwarding_domain = os.environ.get('GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN')
    
    if codespace_name and github_codespaces_port_forwarding_domain:
        url = f"https://{codespace_name}-5000.{github_codespaces_port_forwarding_domain}"
        print(f"\n📡 Your API is accessible at:")
        print(f"\n   {url}")
        print(f"\n🔗 Available endpoints:")
        print(f"   {url}/health")
        print(f"   {url}/predict")
        print(f"   {url}/batch_predict")
        print(f"   {url}/feature_info")
        
        print(f"\n💡 Test in your browser:")
        print(f"   Open: {url}/health")
        
        print(f"\n📝 Or use curl:")
        print(f"   curl {url}/health")
        
        return url
    else:
        print("\n⚠️  Not running in GitHub Codespaces")
        print("   Local access: http://localhost:5000")
        
        print("\n💡 To access in Codespaces:")
        print("   1. Click 'PORTS' tab at the bottom of VS Code")
        print("   2. Port 5000 should be listed")
        print("   3. Click the 'globe' icon or right-click → 'Open in Browser'")
        
        return None

def main():
    """Main function"""
    if check_port_5000():
        url = get_codespace_url()
        
        if url:
            print("\n" + "=" * 60)
            print("✨ QUICK TEST")
            print("=" * 60)
            
            # Test through the public URL
            try:
                req = urllib.request.Request(f"{url}/health")
                with urllib.request.urlopen(req, timeout=10) as response:
                    print(f"\n✅ Public URL is accessible!")
                    data = json.loads(response.read().decode())
                    print(json.dumps(data, indent=2))
            except Exception as e:
                print(f"\n⚠️  Public URL test: {e}")
                print("\n💡 If you see an error, make sure port 5000 is set to 'Public':")
                print("   1. Go to PORTS tab")
                print("   2. Right-click port 5000")
                print("   3. Select 'Port Visibility' → 'Public'")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
