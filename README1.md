<h2>Installation instructions</h2>


<h3>1. Installation of quanser-robots</h3><br />
    clone the repository to some folder:<br /> 
        cd ~<br />
        mkdir tmp<br />
        cd tmp<br />
        git clone https://git.ias.informatik.tu-darmstadt.de/quanser/clients.git<br />

<h3>2. Installation of the virtual environment</h3> <br />
    Take the dependency file and create a new environment from it. You can find further information on creating a virtual environment here: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file<br />
    conda env create -f dependencies.yml<br />
    conda activate group19<br />

    
<h3>3. Installation of the quanser_robots package into the virtual environment</h3><br />
    cd clients<br />
    python3.6 -m pip install -e .<br />
    
<h3>4. Check that everything works correctly</h3>
    python3.6 <br />
    import gym<br />
    import quanser_robots<br />
    env = gym.make('Qube-v0')<br />
    env.reset()<br />
    env.render()<br />




<h2> Algorithms</h2>
<ul>
<li> Natural Actor Critic (NAC)</li>
<li> Deep Deterministic Policy Gradient (DDPG)</li></ul>
