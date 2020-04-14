Now, we will install |miniconda|_ to your account on Hopper.
Log in to Hopper using::

    ssh YOUR-AU-USERNAME@hopper.auburn.edu

or, if you have SSH-tunnelling setup:: 

    ssh hopper

*On hopper*, download the |miniconda|_ installation script::

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

After this command finishes, you should be able to type::

    ls

and see ``Miniconda3-latest-Linux-x86_64.sh`` listed.
This is a shell script that will install |miniconda|_.
To install |miniconda|_ in your Hopper account, run::

    bash Miniconda3-latest-Linux-x86_64.sh

You should see a prompt that looks something like::

    Welcome to Miniconda3 4.8.2
    
    In order to continue the installation process, please review the license
    agreement.
    Please, press ENTER to continue
    >>> 

Hit enter, then scroll down using the space bar or enter key until
you see the following prompt::

    Do you accept the license terms? [yes|no]
    [no] >>> 

Type "yes" and hit enter. Then, you will be prompted to choose the lcoation of
the installation::

    Miniconda3 will now be installed into this location:
    /home/jro0014/miniconda3
    
      - Press ENTER to confirm the location
      - Press CTRL-C to abort the installation
      - Or specify a different location below
    
    [/home/jro0014/miniconda3] >>> 

Simply hit enter to accept the default install location.
After a while (and lots of output), you should see the following prompt::

    Do you wish the installer to initialize Miniconda3
    by running conda init? [yes|no]
    [no] >>> 

Type "yes" and hit enter. This will add some code to your ``.bashrc`` file
that will make using ``conda`` simpler.

If all goes well, you should see::

    Thank you for installing Miniconda3!

at the bottom of the output.

Next, log out of Hopper by entering ``exit`` and then log back in.
After logging back in, update ``conda`` using::

    conda update conda

If there are updates available for ``conda`` you will be prompted with::

    Proceed ([y]/n)? y 

Type "y" (for yes) and hit enter to finish the update.
