// frontend/renderer.js
const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('api', {
  analyzeImage: (imagePath) => ipcRenderer.invoke('analyze-image', imagePath),
  login: (usernameOrEmail, password) => ipcRenderer.invoke('login', usernameOrEmail, password),
  register: (email, username, password) => ipcRenderer.invoke('register', email, username, password),
  logout: () => ipcRenderer.invoke('logout'),
  getCurrentUser: () => ipcRenderer.invoke('get-current-user'),
  getAllUsers: () => ipcRenderer.invoke('get-all-users')
})
