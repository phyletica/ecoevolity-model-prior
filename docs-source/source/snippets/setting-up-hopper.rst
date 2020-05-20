Once you have an account on the Hopper cluster, you can log in from the command
line using (please read the notes below before you try this for the first
time)::

    ssh YOUR-AU-USERNAME@hopper.auburn.edu

.. note:: To log in to Hopper you will need to have Duo 2-factor authentication
    set up.  `Go here <https://duo.auburn.edu/>`_ for how to set up Duo.
    Also, you will need to have your AU 2-factor authentication configured to
    send you a Duo request by default.  Go to https://auburn.edu/2factor and on
    the left side of the screen, click on "My settings & Devices." For the
    "When I log in" setting, choose "Automatically send this device a Duo
    Push."

.. note:: After entering this command, you will get asked to enter the password
   associated with your AU account.

.. note:: The first time you log in you will receive a warning stating
    something like "the authenticiy of the host cannot be established." This is
    normal; it just means that the IP address of the server you are logging
    in to is new and hasn't been added to the ``known_hosts`` file on your
    computer. Simply type ``yes`` and hit enter to log in.

.. note:: This will only work if you are either on AU's campus or are connected
    to the AU virtual private network (VPN).
    `Here's a guide <https://libguides.auburn.edu/vpn>`_
    for how to set up the VPN so that you can log in from off campus.

To log out of Hopper, simply type::

    exit


Setting up passwordless login
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can log in to Hopper more securely and without needing to enter our AU
password.
To do this, we will use SSH authentication keys.
To see if you have SSH keys on your machine (e.g., your laptop),
open your terminal and enter::

    ls ~/.ssh

If you see ``id_rsa`` and ``id_rsa.pub`` listed, you already have a pair of SSH
keys.
If you don't see these listed (or the ``.ssh`` directory does not exist),
you need to create a pair of SSH keys.
To do this, we will use ``ssh-keygen``.
When you enter this on the command line, it will prompt you with some options;
simply hit enter for each prompt to accept the default (note, if you already
have a pair of SSH keys, skip this step)::

    ssh-keygen
    
Now, when you::

    ls ~/.ssh

You should see the files ``id_rsa`` and ``id_rsa.pub`` listed.

``id_rsa``
    This is the private component of your SSH key pair; do not share it with
    anyone.

``id_rsa.pub``
    This is the public component of your SSH key pair and can be safely shared.

Now, we need to put the contents of ``id_rsa.pub`` into the
``.ssh/authorized_keys`` file in your home directory on Hopper.
We can do this with one command entered on *your* computer::

    cat ~/.ssh/id_rsa.pub | ssh YOUR-AU-USERNAME@hopper.auburn.edu "cat >> ~/.ssh/authorized_keys"

Now, after you start a fresh shell session, you should be able to log in to
Hopper without entering your password.


Setting up alternative to VPN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Everything above about setting up Hopper assumes you were either on AU's
campus, or were connected to the AU VPN.
AU's VPN is ... let's just say, not always great.
If you are having trouble with the VPN and need to access Hopper from
off campus, we can set up a workaround.
To do this you will need an account on AU's Venus Linux server.

Once you have an account on Venus, you can log in via::

    ssh YOUR-AU-USERNAME@venus2.auburn.edu

When logged in to Venus, check to see if you have SSH keys::

    ls ~/.ssh

If you see ``id_rsa`` and ``id_rsa.pub`` listed, you already have a pair of SSH
keys.
If you don't see these listed (or the ``.ssh`` directory does not exist),
you need to create a pair of SSH keys, like we did above, 
using ``ssh-keygen``.
Enter (make sure you are logged in to Venus for this)::

    ssh-keygen
    
and hit enter at all the prompts to accept the defaults.
Now, when you::

    ls ~/.ssh

You should see the files ``id_rsa`` and ``id_rsa.pub`` listed.
Now, we need to add the content of your public SSH key on Venus 
to your ``authorized_keys`` file on Hopper (again, make sure you are logged
in to Venus for this)::

    cat ~/.ssh/id_rsa.pub | ssh YOUR-AU-USERNAME@hopper.auburn.edu "cat >> ~/.ssh/authorized_keys"

Now, log out of Venus using::

    exit

so that you are back on *your* computer.
Now, we need to add the content of your public SSH key on *your* computer to
the ``authorized_keys`` file on Venus (make sure you run this command on your
computer)::

    cat ~/.ssh/id_rsa.pub | ssh YOUR-AU-USERNAME@venus2.auburn.edu "cat >> ~/.ssh/authorized_keys"

Next, we need to set up an SSH configuration file on *your* computer.
So, on your machine enter::

    touch ~/.ssh/config

This will create the file ``~/.ssh/confg`` if it doesn't exist, and does
nothing if it already exists.
Next, open the file ``~/.ssh/config`` with a "raw" text editor.
One option is to edit this file using ``nano``::

    nano ~/.ssh/config

Add the following content to the file, replacing ``YOUR-AU-USERNAME``
with your AU username::

    Host venus 
        HostName    venus2.auburn.edu
        User        YOUR-AU-USERNAME
    Host hopper
        HostName    hopper.auburn.edu
        User        YOUR-AU-USERNAME
        ProxyJump   venus

After saving this content to your SSH config file, and starting a fresh shell session,
you should be able to log in to Hopper using::

    ssh hopper

This command will use "SSH tunnelling" to log you in to Hopper by
using Venus as an intermediary.
Because Venus is accessible off-campus, this login should work on and off
campus, regardless of whether you are connected to the AU VPN.


Create your scratch directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will be conducting the analyses for this project from the ``/scratch``
storage on hopper.
There is a huge amount of (fast) hard drive space mounted at the ``/scratch``
directory to which all Hopper users have access.
Make your own directory in ``/scratch`` using (make sure you are logged in to
Hopper for this)::

    mkdir /scratch/YOUR-AU-USERNAME
