using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;
using System.IO;

public class PythonLauncher : MonoBehaviour
{
    private Process pythonProcess;

    // Unity가 시작될 때 Python 서버 실행
    void Start()
    {
        string pythonPath = Path.Combine(Application.streamingAssetsPath, "Python/FTWhisperKM.exe");
        string workingDir = Path.Combine(Application.streamingAssetsPath, "Python/");

        UnityEngine.Debug.Log($"Python Path: {pythonPath}");
        UnityEngine.Debug.Log($"Working Directory: {workingDir}");

        pythonProcess = new Process();
        pythonProcess.StartInfo.FileName = pythonPath;
        pythonProcess.StartInfo.WorkingDirectory = workingDir;
        //콘솔 창 나타냄: UseShellExecute = true, CreatNoWindow = false
        //콘솔 창 숨김: UseShellExecute = false, CreatNoWindow = true
        pythonProcess.StartInfo.UseShellExecute = true;
        pythonProcess.StartInfo.CreateNoWindow = false;
        pythonProcess.Start();

        UnityEngine.Debug.Log("Python 실행 파일이 실행되었습니다.");
    }

    // Unity가 종료될 때 Python 서버 종료
    void OnApplicationQuit()
    {
        if (pythonProcess != null && !pythonProcess.HasExited)
        {
            pythonProcess.Kill();
            UnityEngine.Debug.Log("Python 실행 파일이 종료되었습니다.");
        }
    }
}
