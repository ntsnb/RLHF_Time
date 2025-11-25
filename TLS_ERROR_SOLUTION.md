# TLS连接错误解决方案

## 问题描述
```
fatal: 无法访问 'https://github.com/ntsnb/RLHF_Time.git/'：gnutls_handshake() failed: The TLS connection was non-properly terminated.
```

## 解决方案

### 解决方案1：配置Git使用HTTP而不是HTTPS
```bash
git config --global http.sslVerify false
git remote set-url origin http://github.com/ntsnb/RLHF_Time.git
git push -u origin main
```

### 解决方案2：重新配置远程地址（推荐）
```bash
git remote remove origin
git remote add origin https://github.com/ntsnb/RLHF_Time.git
git push -u origin main
```

### 解决方案3：使用SSH密钥（更安全）
1. 生成SSH密钥：
```bash
ssh-keygen -t ed25519 -C "您的邮箱"
```

2. 将SSH密钥添加到SSH agent：
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

3. 复制公钥并添加到GitHub：
```bash
cat ~/.ssh/id_ed25519.pub
```

4. 复制公钥内容，在GitHub设置中添加SSH密钥

5. 使用SSH地址推送：
```bash
git remote set-url origin git@github.com:ntsnb/RLHF_Time.git
git push -u origin main
```

### 解决方案4：更新系统证书（如果是Ubuntu系统）
```bash
sudo apt update
sudo apt install ca-certificates
```

## 推荐步骤

按照以下顺序尝试：

1. **先尝试方案1**：
```bash
git config --global http.sslVerify false
git push -u origin main
```

2. **如果失败，使用方案2**：
```bash
git remote remove origin
git remote add origin https://github.com/ntsnb/RLHF_Time.git
git push -u origin main
```

3. **如果还是失败，尝试方案4**然后重试

4. **长期解决方案：配置SSH密钥（方案3）**

## 验证连接

可以先测试GitHub连接：
```bash
ssh -T git@github.com
```

如果是SSH配置成功，应该看到类似：
```
Hi ntsnb! You've successfully authenticated, but GitHub does not provide shell access.