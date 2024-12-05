using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class NetworkManager
{
    private TcpListener listener;
    private Thread serverThread;
    private readonly Queue<string> commandQueue;

    public NetworkManager(Queue<string> sharedQueue)
    {
        commandQueue = sharedQueue;
    }

    public void StartServer()
    {
        serverThread = new Thread(ServerLoop);
        serverThread.Start();
    }

    private void ServerLoop()
    {
        listener = new TcpListener(System.Net.IPAddress.Parse("127.0.0.1"), 5005);
        listener.Start();
        Debug.Log("Server started, waiting for connections...");

        while (true)
        {
            using (TcpClient client = listener.AcceptTcpClient())
            {
                // 클라이언트 연결 시 메시지 출력
                Debug.Log("Client connected!");

                NetworkStream stream = client.GetStream();
                byte[] buffer = new byte[1024];
                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                string command = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                Debug.Log($"Received command: {command}");

                lock (commandQueue)
                {
                    commandQueue.Enqueue(command);
                }
            }
        }
    }

    public void StopServer()
    {
        listener?.Stop();
        serverThread?.Abort();
    }
}