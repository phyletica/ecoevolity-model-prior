.. _installing-git:

Installing Git
^^^^^^^^^^^^^^

The system-wide version of |git|_ on Hopper is too old for our needs, and the
version of |git|_ that is available as a module was not built correctly.
So, each of us will install our own version of |git|_ to our home directory on
Hopper.

Log in to Hopper using::

    ssh YOUR-AU-USERNAME@hopper.auburn.edu

or, if you have SSH-tunnelling setup:: 

    ssh hopper

To avoid library conflicts, let's deactivate ``conda`` before we build
``git``::

    conda deactivate

*On hopper*, download an archive of the |git|_ source code::

    wget https://mirrors.edge.kernel.org/pub/software/scm/git/git-2.26.2.tar.gz

After this command finishes, extract the archive::

    tar xzf git-2.26.2.tar.gz

and then ``cd`` into the directory with the |git|_ source code::

    cd git-2.26.2

Compile the source code using::

    make

This will build the ``git`` executables from the source code.
This will take a few minutes.
When it finishes, install the ``git`` executables into a ``bin``
directory within your Hopper home directory using::

    make install

After it finishes, you can ``cd`` back out of the directory with the |git|_
source code::

    cd ..

and then cleanup by removing everything you downloaded and extracted::

    rm -r git-2.26.2*

Next, you need to add the ``bin`` directory in your home folder to your PATH
variable so that Bash knows where to find the ``git`` executable when you type
``git`` at the command line.
To do this, we will use ``echo`` and redirection (``>>``) to append a knew line
to the hidden Bash configuration file called ``.bashrc``::

    echo 'export PATH="${HOME}/bin:${PATH}"' >> ~/.bashrc

Now, log off Hopper with ``exit`` and then log back in.
After logging back in, when you type::

    which git

you should get the output::

    ~/bin/git

and when you enter::

    git --version

you should see::

    git version 2.26.2


.. _configuring-git:

Configuring Git
^^^^^^^^^^^^^^^

Next, we will set some configuration settings for ``git``.
The latest versions of ``git`` issue an annoying warning when
you ``git pull``.
Historically, the ``pull`` command was simply a shortcut to ``git fetch``
followed by ``git merge``.
However, you can also replace the ``merge`` step with ``rebase``.
We do **not** want to use ``rebase``, so we will tell ``git``
to use ``merge`` when we ``pull``::

    git config --global pull.rebase false

If you haven't configured ``git`` on Hopper with your name and email, go ahead
and do that now using the following 2 commands (just change my name and email
to yours, and **use the email associated with your** |github|_ **account**)::

    git config --global user.name "Jamie R. Oaks"
    git config --global user.email "joaks1@gmail.com"

If you have an account on GitHub, give Git your GitHub username too (again,
change my username to yours)::

    git config --global github.user "joaks1"

When you ``commit`` content to a git repository, git will open a text editor
for you to write a message to accompany your commit.
By default, it will open ``vim``. So, if you are not confortable using ``vim``
(if you don't know what ``vim`` is, then you won't be comfortable using it),
then configure ``git`` to open ``nano`` instead::

    git config --global core.editor "nano -w"

Next, tell ``git`` to use pretty colors when it gives you messages::

    git config --global color.ui "true"


.. _installing-git-lfs:

Installing git-lfs
^^^^^^^^^^^^^^^^^^

|git|_ is great at tracking the contents of "raw" text files.
In this project, we will produce some large, compressed archives of files.
We do not want |git|_ to try to track the contents of these files, because it
will slow things down a lot, and blow up the size of the |git|_ database in our
project.
Luckily, there is a really nice extension for |git|_ for handling large files
called `Git Large File Storage (LFS) <https://git-lfs.github.com/>`_.
This extension allows us to add large files to out git repo, but prevents
``git`` from trying to track the content of these files "line-by-line".

So, let's install Git LFS.
Use the next three commands to navigate to your home directory, create a new
directory called ``git-lfs``, and then ``cd`` into it::

    cd ~

Next, download an archive of the Git LFS software using ``wget``::

    wget https://github.com/git-lfs/git-lfs/releases/download/v2.10.0/git-lfs-linux-amd64-v2.10.0.tar.gz

When the download finishes, make a new directory called ``git-lfs-files`` and
then extract the files from the archive into the this new directory::

    mkdir git-lfs-files
    tar xzf git-lfs-linux-amd64-v2.10.0.tar.gz -C git-lfs-files

Then, ``cd`` into the ``git-lfs-files`` directory::

    cd git-lfs-files

Now, set the ``PREFIX`` variable to tell the installation script where
to put ``git-lfs``, and then run the installation script::

    export PREFIX="$HOME"
    bash install.sh

Move back out of the ``git-lfs`` and cleanup by removing everything
we downloaded and extracted::

    cd ..
    rm -r git-lfs*

Lastly, run the following command to configure ``git-lfs`` for your Hopper
account::

    git lfs install --skip-smudge
