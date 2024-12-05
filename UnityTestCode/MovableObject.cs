using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovableObject : MonoBehaviour
{
    [HideInInspector] // Inspector 창에 표시하지 않음
    public string objectName;

    private void Awake()
    {
        // Unity 오브젝트의 이름을 가져와 저장
        objectName = gameObject.name;
    }

    public void Move(Vector3 direction)
    {
        transform.Translate(direction);
        Debug.Log($"{objectName} moved: {direction}");
    }
}
