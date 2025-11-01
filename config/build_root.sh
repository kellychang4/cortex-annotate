#! /bin/bash
################################################################################
# build_root.sh
# This file is a placeholder for any commands that should be run when the docker
# image for the annotate tool is built. The commands in this file are run by the
# root user inside the docker-image after all other parts of the build process
# except for the build_user.sh script, also in this directory.
# You do not need to do anything to this script, but if you need to install
# something or configure something during the docker build process, you can put
# the required commands in this file instead of editing the Dockerfile directly.


architecture=$( uname -m )
case "${architecture}" in 
  x86_64)  architecture="amd64" ;; 
  aarch64) architecture="arm64" ;; 
esac 

filename="git-annex-standalone-${architecture}.tar.gz"
wget -P /opt "https://downloads.kitenet.net/git-annex/linux/current/${filename}" && \
  tar -xf "/opt/${filename}" -C /opt && \
  ln -s /opt/git-annex.linux/git-annex /usr/local/bin/git-annex && \
  chmod +x /usr/local/bin/git-annex && \
  rm -r "/opt/${filename}"

cmd='\n\n# Prepare Datalad dataset\n'
cmd+='INIT_DATASET="/data/studyforrest"\n'
cmd+='DEST_DATASET="/cache/studyforrest"\n\n'
cmd+='# Check if the destination directory is uninitialized (no .git folder)\n'
cmd+='if [ ! -d "${DEST_DATASET}/.git" ]; then\n'
cmd+='  cp -r "${INIT_DATASET}" "${DEST_DATASET}"\n'
cmd+='fi'

sed -i "s,set -e,set -e${cmd}," /usr/local/bin/start-notebook.sh
