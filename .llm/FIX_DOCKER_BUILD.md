# Fix Docker Build Permission Error

## Quick Fix

```bash
# Option 1: Fix permissions on existing directory
cd ~
sudo chown -R $USER:$USER llamafactory-fp8-hopper
chmod -R 755 llamafactory-fp8-hopper
cd llamafactory-fp8-hopper

# Option 2: Fresh clone (recommended)
cd ~
rm -rf llamafactory-fp8-hopper
git clone https://github.com/sbhavani/llamafactory-fp8-hopper.git
cd llamafactory-fp8-hopper

# Now build (no sudo needed for docker build)
docker build -t llamafactory-fp8:latest .

# If you still need sudo for docker
sudo docker build -t llamafactory-fp8:latest .
```

---

## Root Cause

The error occurs because:
1. **Directory permissions**: Docker daemon can't access the build context
2. **User mismatch**: Directory might be owned by root or another user
3. **Symlink issues**: The path might contain problematic symlinks

---

## Detailed Steps

### 1. Check Current Permissions
```bash
ls -la ~/llamafactory-fp8-hopper
```

### 2. Fix Ownership
```bash
# Make sure you own the directory
sudo chown -R $USER:$USER ~/llamafactory-fp8-hopper

# Fix permissions
chmod -R 755 ~/llamafactory-fp8-hopper
```

### 3. Verify Directory Contents
```bash
cd ~/llamafactory-fp8-hopper
ls -la
# Should see: Dockerfile, configs/, scripts/, etc.
```

### 4. Build Without sudo (Preferred)
```bash
# Add yourself to docker group (if not already)
sudo usermod -aG docker $USER
newgrp docker  # Or logout/login

# Build without sudo
docker build -t llamafactory-fp8:latest .
```

### 5. Build With sudo (If needed)
```bash
sudo docker build -t llamafactory-fp8:latest .
```

---

## Alternative: Clone Fresh

If permissions are messy, cleanest solution:

```bash
# Remove old directory
cd ~
rm -rf llamafactory-fp8-hopper

# Clone fresh from GitHub
git clone https://github.com/sbhavani/llamafactory-fp8-hopper.git

# Enter directory
cd llamafactory-fp8-hopper

# Build
docker build -t llamafactory-fp8:latest .
```

---

## If Still Failing

### Check Docker Daemon Permissions
```bash
# Check if docker is running
sudo systemctl status docker

# Check your user is in docker group
groups $USER | grep docker

# If not in docker group, add and reboot
sudo usermod -aG docker $USER
sudo reboot
```

### Check SELinux (RHEL/CentOS)
```bash
# Check if SELinux is causing issues
getenforce

# If enforcing, try temporarily disabling
sudo setenforce 0

# Build again
docker build -t llamafactory-fp8:latest .

# Re-enable SELinux
sudo setenforce 1
```

### Alternative Build Location
```bash
# Try building from /tmp
sudo cp -r ~/llamafactory-fp8-hopper /tmp/
cd /tmp/llamafactory-fp8-hopper
sudo docker build -t llamafactory-fp8:latest .
```

---

## Expected Build Output

When successful, you should see:

```
[+] Building 45.2s (12/12) FINISHED
=> [internal] load build definition from Dockerfile
=> => transferring dockerfile: 1.23kB
=> [internal] load .dockerignore
=> => transferring context: 2B
=> [internal] load metadata for nvcr.io/nvidia/pytorch:25.10-py3
...
=> exporting to image
=> => exporting layers
=> => writing image sha256:...
=> => naming to docker.io/library/llamafactory-fp8:latest
```

---

## Verify Image Built
```bash
docker images | grep llamafactory-fp8
# Should show: llamafactory-fp8  latest  ...  XX GB
```
