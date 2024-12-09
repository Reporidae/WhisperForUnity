using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;
using System.IO;

public class PythonLauncher : MonoBehaviour
{
    private Process pythonProcess;

    // Unity�� ���۵� �� Python ���� ����
    void Start()
    {
        string pythonPath = Path.Combine(Application.streamingAssetsPath, "Python/FTWhisperKM.exe");
        string workingDir = Path.Combine(Application.streamingAssetsPath, "Python/");

        UnityEngine.Debug.Log($"Python Path: {pythonPath}");
        UnityEngine.Debug.Log($"Working Directory: {workingDir}");

        pythonProcess = new Process();
        pythonProcess.StartInfo.FileName = pythonPath;
        pythonProcess.StartInfo.WorkingDirectory = workingDir;
        //�ܼ� â ��Ÿ��: UseShellExecute = true, CreatNoWindow = false
        //�ܼ� â ����: UseShellExecute = false, CreatNoWindow = true
        pythonProcess.StartInfo.UseShellExecute = true;
        pythonProcess.StartInfo.CreateNoWindow = false;
        pythonProcess.Start();

        UnityEngine.Debug.Log("Python ���� ������ ����Ǿ����ϴ�.");
    }

    // Unity�� ����� �� Python ���� ����
    void OnApplicationQuit()
    {
        if (pythonProcess != null && !pythonProcess.HasExited)
        {
            pythonProcess.Kill();
            UnityEngine.Debug.Log("Python ���� ������ ����Ǿ����ϴ�.");
        }
    }
}
