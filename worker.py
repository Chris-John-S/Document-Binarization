import paramiko


def main():
    try:
        # Create an SSH server (worker node)
        ssh_server = paramiko.Transport(('0.0.0.0', 22))

        # Replace 'tory' and '11' with your actual username and password
        ssh_server.add_server_key(paramiko.RSAKey.generate(2048))
        ssh_server.set_subsystem_handler('sftp', paramiko.SFTPServer, paramiko.SFTPServerInterface)

        # Start the SSH server on the worker node
        ssh_server.start_server()

        print("Worker node is ready.")

        # Keep the worker node running until manually stopped
        while True:
            pass

    except Exception as e:
        print(f"Error on worker node: {str(e)}")


if __name__ == '__main__':
    main()
