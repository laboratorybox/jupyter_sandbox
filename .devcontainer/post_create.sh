# #!/bin/bash

# set -ex

# sudo usermod -aG docker vscode
# newgrp docker

# # reactopya
cd /workspaces/reactopya
pip install --no-deps -e .
echo "export PATH=/workspaces/reactopya/bin:\$PATH" >> ~/.bashrc

# hither2
cd /workspaces/hither2
pip install --no-deps -e .
echo "export PATH=/workspaces/hither2/bin:\$PATH" >> ~/.bashrc

# kachery
cd /workspaces/kachery
pip install --no-deps -e .
echo "export PATH=/workspaces/kachery/bin:\$PATH" >> ~/.bashrc

# spikeforest2
cd /workspaces/spikeforest2
pip install --no-deps -e .


