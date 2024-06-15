import rpyc

## Only needed on Server Side
from rpyc.utils.server import ThreadedServer


# Class GreetingService defines say_hello capability.
# Any process hosting this service, will expose the say_hello
# API over rpc on host localhost over port 8082
@rpyc.service
class GreetingService(rpyc.Service):
    @rpyc.exposed
    def say_hello(self, user: str):
        print("say_hello is called")
        return f"Hello {user}!"


# Start a server
server = ThreadedServer(GreetingService, port=8082)
server.start()


## On client side
# connect to the server and invoke remote API as if it was local
connection = rpyc.connect("localhost", 18811)
print(connection.root.say_hello("Jo"))
print(connection.root.say_hello("Jo"))
