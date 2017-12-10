#!/usr/bin/expect

# Enters the password to the ssh session.

set cmd [lrange $argv 1 end]
set password [lindex $argv 0]

eval spawn $cmd
expect "*cgcontact*assword:"
send "$password\r";

expect "*cgpool*assword:"
send "$password\r";

sleep 1
interact