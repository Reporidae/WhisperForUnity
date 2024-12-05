using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CommandProcessor
{
    private readonly Dictionary<string, MovableObject> objects;

    public CommandProcessor(List<MovableObject> objectList)
    {
        objects = new Dictionary<string, MovableObject>();

        foreach (var obj in objectList)
        {
            // objectName (즉, GameObject.name)을 키로 등록
            objects[obj.objectName] = obj;
        }
    }

    public void ProcessCommand(string command)
    {
        // 명령어를 "오브젝트|행동"으로 분리
        string[] parts = command.Split('|');
        if (parts.Length < 2)
        {
            Debug.Log("Invalid command format");
            return;
        }

        string targetName = parts[0]; // 예: "MAGE"
        string action = parts[1];     // 예: "MOVE_FORWARD"

        // 대상 오브젝트 찾기
        if (objects.TryGetValue(targetName, out MovableObject targetObject))
        {
            switch (action)
            {
                case "MOVE_FORWARD":
                    targetObject.Move(Vector3.forward);
                    break;
                case "MOVE_BACKWARD":
                    targetObject.Move(Vector3.back);
                    break;
                case "MOVE_RIGHT":
                    targetObject.Move(Vector3.right);
                    break;
                case "MOVE_LEFT":
                    targetObject.Move(Vector3.left);
                    break;
                case "MOVE_UP":
                    targetObject.Move(Vector3.up);
                    break;
                case "MOVE_DOWN":
                    targetObject.Move(Vector3.down);
                    break;
                default:
                    Debug.Log($"Unknown action: {action}");
                    break;
            }
        }
        else
        {
            Debug.Log($"Target object not found: {targetName}");
        }
    }
}