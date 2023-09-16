import os
import sys
import subprocess
import re
from shutil import which
def is_command(cmd):
    return which(cmd) is not None
    
def check_amd_gpu_lspci():
    check_cmd ="lspci"
    try:
        ps1 = subprocess.run(check_cmd.split(),  stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT, check=True)
        for line in str.splitlines(ps1.stdout.decode('utf-8')):
            if re.search(r'ATI] Device 740f', line):   # MI210
                return True
            elif re.search(r'ATI] Device 740c', line): # MI250 
                return True
            elif re.search(r'ATI] Aldebaran', line): # MI200
                return True
            elif re.search(r'ATI] Device 7408', line): # MI250x
                return True
            elif re.search(r'ATI] Device 738c', line): # MI100
                return True  
            elif re.search(r'ATI] Device Arcturus', line): # MI200
                return True
            elif re.search(r'ATI] Device 66af', line): # MI50
                return True                              
        return False
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        return False
def check_amd_gpu_rocminfo():
    check_cmd ="rocminfo"
    try:
        ps1 = subprocess.run(check_cmd.split(),  stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT, check=True)
        for line in str.splitlines(ps1.stdout.decode('utf-8')):
            if re.search(r'gfx906', line): # MI50/MI60
                return True
            elif re.search(r'gfx908', line): # MI100
                return True
            elif re.search(r'gfx90a', line): # MI200
                return True
        return False
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        return False
def check_rocm_packages( ):
    UBUNTU_TYPE = "ubuntu"	
    DEBIAN_TYPE = "debian"	
    RHEL_TYPE = "rhel"	    
    CENTOS_TYPE = "centos"	
    SLES_TYPE = "sles"	    
    PKGTYPE_RPM = "rpm"
    PKGTYPE_DEB = "deb"
    RPM_CMD = "/usr/bin/rpm"
    DPKG_CMD = "/usr/bin/dpkg"
    try:
        import distro
        linux_id =  distro.id()
    except ModuleNotFoundError as err:
        ETC_OS_RELEASE = "/etc/os-release"
        with open(ETC_OS_RELEASE, 'r') as f:
            for line in f:
                if CENTOS_TYPE.lower() in line.lower():
                    linux_id = CENTOS_TYPE
                    break
                if DEBIAN_TYPE.lower() in line.lower():
                    linux_id = UBUNTU_TYPE
                    break
                if UBUNTU_TYPE.lower() in line.lower():
                    linux_id = UBUNTU_TYPE
                    break
                if SLES_TYPE.lower() in line.lower():
                    linux_id= SLES_TYPE
                    break
                if RHEL_TYPE.lower() in line.lower():
                    linux_id = RHEL_TYPE
                    break       
    pkgtype = {
        CENTOS_TYPE : PKGTYPE_RPM,
        RHEL_TYPE : PKGTYPE_RPM,
        CENTOS_TYPE : PKGTYPE_RPM,
        UBUNTU_TYPE : PKGTYPE_DEB,
        SLES_TYPE : PKGTYPE_RPM
    }[linux_id]
    if pkgtype is PKGTYPE_RPM:
        check_cmd = RPM_CMD + " -q rock-dkms"
        try:
            ps1 = subprocess.run(check_cmd.split(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=True)
            return True
        except subprocess.CalledProcessError as err:
            pass
        check_cmd = RPM_CMD + " -q amdgpu-dkms"
        try:
            ps1 = subprocess.run(check_cmd.split(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=True)
            return True
        except subprocess.CalledProcessError as err:
            pass
        check_cmd = RPM_CMD + " -q hip-devel"
        try:
            ps1 = subprocess.run(check_cmd.split(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=True)
            return True
        except subprocess.CalledProcessError as err:
            return False
    elif pkgtype is PKGTYPE_DEB:
        check_cmd = DPKG_CMD + " -l rock-dkms"
        try:
            ps1 = subprocess.run(check_cmd.split(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=True)
            for line in str.splitlines(ps1.stdout.decode('utf-8')):
                if re.search(r'^i.*rock-dkms.*all', line): # 'i' for installed
                    return True
        except subprocess.CalledProcessError as err:
            pass
        check_cmd = DPKG_CMD + " -l amdgpu-dkms"
        try:
            ps1 = subprocess.run(check_cmd.split(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=True)
            for line in str.splitlines(ps1.stdout.decode('utf-8')):
                if re.search(r'^i.*amdgpu-dkms.*all', line): # 'i' for installed
                    return True
        except subprocess.CalledProcessError as err:
            pass
        check_cmd = DPKG_CMD + " -l miopen-hip"
        try:
            ps1 = subprocess.run(check_cmd.split(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=True)
            for line in str.splitlines(ps1.stdout.decode('utf-8')):
                if re.search(r'^i.*miopen-hip.*', line): # 'i' for installed
                    return True
        except subprocess.CalledProcessError as err:
            pass
        check_cmd = DPKG_CMD + " -l hip-dev"
        try:
            ps1 = subprocess.run(check_cmd.split(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=True)
            for line in str.splitlines(ps1.stdout.decode('utf-8')):
                if re.search(r'^i.*hip-dev.*', line): # 'i' for installed
                    return True
        except subprocess.CalledProcessError as err:
            return False
    else:
        print("Unknown package type {}. Cannot detect rock-dkms amdgpu-dkms status.".format(pkgtype))
        return False    
