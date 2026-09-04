import subprocess
import sys

def main():
    """
    启动论文阅读助手
    """
    print("正在启动智能论文阅读助手...")
    try:
        # 启动client.py
        result = subprocess.run([sys.executable, "client.py"], check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"启动失败: {e}")
        return e.returncode
    except FileNotFoundError:
        print("错误: 找不到client.py文件")
        return 1

if __name__ == "__main__":
    sys.exit(main())