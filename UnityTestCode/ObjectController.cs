using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjectController : MonoBehaviour
{
    private NetworkManager networkManager;
    private CommandProcessor commandProcessor;
    private readonly Queue<string> commandQueue = new Queue<string>();

    void Start()
    {
        // ��� MovableObject�� �˻��Ͽ� CommandProcessor�� ����
        MovableObject[] movableObjects = FindObjectsByType<MovableObject>(FindObjectsSortMode.None);
        commandProcessor = new CommandProcessor(new List<MovableObject>(movableObjects));

        networkManager = new NetworkManager(commandQueue);
        networkManager.StartServer();
    }

    void Update()
    {
        lock (commandQueue)
        {
            while (commandQueue.Count > 0)
            {
                string command = commandQueue.Dequeue();
                commandProcessor.ProcessCommand(command);
            }
        }
    }

    private void OnApplicationQuit()
    {
        if (networkManager != null)
        {
            networkManager.StopServer();
        }
    }
}
