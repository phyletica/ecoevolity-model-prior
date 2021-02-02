If you don't already have an account on |github|_,
go there and sign up for a free account.
Log in to your |github|_ account, and go to the "Settings" for your account
(there should be a drop down near the top right corner of the |github|_ page,
with a "Settings" option).
Along the left side of your settings page, there should be a "SSH and GPG
keys" link; click on this.

Open a shell (terminal) session on your computer and enter::

    cat ~/.ssh/id_rsa.pub

The output of this command will show your public SSH key, which should look
something like::

    ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCaZow6ifUAg3g7Qj7N5zJ5fMnQoP
    mpAhUqwsvHu/BoXH17TcqP4HdcNoDcprVRwAJL/6ECabdzDyUlGThMKB8w3APqQjqa
    7yc98ymdK0LXDpU0IuWCktW9pyn72XizE1bSIOhDkrFKGmtpLW/0jxGhcRN6OtCUI4
    V98c2AuU3RKZeTSgfEAWnPJcCQaJkvzktVXO55rsNRe6UxqV4B6O/29YhBeCqyLDL6
    VUa7hT+4cqVX8gjLjgDq8jWwxkgeifEt9G1j41 jamie@jamie-XPS-13-9350

Copy the content of this output.
Now, go back to GitHub and click "New SSH key."
Paste your copied SSH key into the "Key" field on GitHub.
Enter a "Title" that will help you remember what computer the key is from
(e.g., xps-laptop).
Then hit "Add SSH Key."

Next, we will log in to Hopper and repeat the same steps.
So, log in to Hopper using::

    ssh YOUR-AU-USERNAME@hopper.auburn.edu

or, if you have SSH-tunnelling setup:: 

    ssh hopper

Now, on *Hopper* ``cat`` the contents of your public SSH key *on Hopper*::

    cat ~/.ssh/id_rsa.pub

Copy the output SSH key, click "New SSH key" on GitHub, and paste your SSH key
into the "Key" field, add a meaningful name into the "Title" field (e.g.,
Hopper), and click "Add SSH Key."
