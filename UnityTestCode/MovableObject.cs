using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovableObject : MonoBehaviour
{
    [HideInInspector] // Inspector â�� ǥ������ ����
    public string objectName;

    private void Awake()
    {
        // Unity ������Ʈ�� �̸��� ������ ����
        objectName = gameObject.name;
    }

    public void Move(Vector3 direction)
    {
        transform.Translate(direction);
        Debug.Log($"{objectName} moved: {direction}");
    }
}
