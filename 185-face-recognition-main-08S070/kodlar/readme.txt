Step 1. 啟動資料夾內 vs_buildtools.exe 
    搜尋 C++ CMake tools for Windows 並安裝


Step 2. 依據 Windows 版號安裝 Windows SDK
    ** winsdksetup.exe 是 SDK 19041版 
    (不一定是這個，須依據windows版號決定，可以上網搜尋"Windows xxx版 SDK"，也可以透過Visual Studio Installer安裝)
    
    備註：
    Windows 10 21H1 版並未引進新的 API，所以此版本將不會隨附發行新的 Windows SDK。
    仍應繼續使用 Windows 10 2004 版的 Windows 10 SDK。( 19041 仍然是最新的目標版本)


Step 3. 下載 CUDA ToolKit & cuDNN
    (如果想用 conda install tensorflow-gpu 就可以跳過這個步驟)
    依據自己的 CUDA Driver version 下載並安裝 CUDA ToolKit
    再依據 CUDA ToolKit version 下載 cuDNN 壓縮檔
    解開壓縮檔後，將include、bin、lib三個資料夾複製到：
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4 底下(不一定是v11.4看版本號)
    
    到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\bin 底下找到 cusolver64_11.dll
    複製一份並改名為 cusolver64_10.dll
    
    接著將 include、bin、lib 三個路徑加入環境變數，重開機。


Step 4. conda 建立虛擬環境，並安裝套件。
    conda create --name xxx python=3.7
    conda activate xxx
    
    有做Step3.的話，pip install tensorflow-gpu==1.13.1
    沒做Step3.的話，conda install tensorflow-gpu=1.13.1
    
    接著，pip install -r requirement.txt


Step 5. 下載對應的PyQt4 (需要根據python版本下載)
    https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqt4
    下載的檔案為.whl檔，安裝方式：pip install PyQt4-4.11.4-cp37-cp37m-win_amd64.whl


Step6. 出現 "DLL load failed: 找不到指定的模組" 的錯誤訊息
    pip uninstall tensorflow-gpu
    pip install rensorflow-gpu==1.13.1