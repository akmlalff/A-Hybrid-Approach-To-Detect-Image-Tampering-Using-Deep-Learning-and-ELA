// main.js
const { app, BrowserWindow, ipcMain } = require('electron')
const path = require('path')
const { spawn } = require('child_process')
const { initDatabase, registerUser, loginUser, getAllUsers } = require('./backend/database')

// Store the current user session
let currentUser = null;

const userDataPath = app.getPath('userData');

let db;

async function createWindow() {
  await initDatabase(userDataPath);
  
  const win = new BrowserWindow({
    width: 900, height: 700,
    webPreferences: {
      preload: path.join(__dirname, 'frontend', 'renderer.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  })
  
  // Check if user is logged in
  if (currentUser) {
    win.loadFile(path.join(__dirname, 'frontend', 'index.html'))
  } else {
    win.loadFile(path.join(__dirname, 'frontend', 'login.html'))
  }
}

app.whenReady().then(createWindow)

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})

// Authentication handlers
ipcMain.handle('register', async (_, email, username, password) => {
  try {
    const user = await registerUser(email, username, password);
    return user;
  } catch (error) {
    throw new Error(error.message);
  }
})

ipcMain.handle('login', async (_, usernameOrEmail, password) => {
  try {
    const user = await loginUser(usernameOrEmail, password);
    currentUser = user;
    return user;
  } catch (error) {
    throw new Error(error.message);
  }
})

ipcMain.handle('logout', async () => {
  currentUser = null;
  return true;
})

ipcMain.handle('get-current-user', async () => {
  if (!currentUser) {
    throw new Error('Not logged in');
  }
  return currentUser;
})

// Admin handlers
ipcMain.handle('get-all-users', async () => {
  if (!currentUser || !currentUser.is_admin) {
    throw new Error('Admin access required');
  }
  
  try {
    return await getAllUsers();
  } catch (error) {
    throw new Error(error.message);
  }
})

// Image analysis handler
ipcMain.handle('analyze-image', async (_, imagePath) => {
  // Check if user is logged in
  if (!currentUser) {
    throw new Error('Authentication required');
  }
  
  return new Promise((resolve, reject) => {
    let stdout = '', stderr = ''
    const py = spawn(path.join(process.resourcesPath, 'inference.exe'), [
      '--image', imagePath
    ])

    py.stdout.on('data', data => { stdout += data })
    py.stderr.on('data', data => { stderr += data })

    py.on('close', code => {
      console.log('--- Python exited with code', code, '---')
      console.log('STDOUT:', stdout)
      console.log('STDERR:', stderr)

      if (code === 0) {
        try {
          const result = JSON.parse(stdout)
          resolve(result)
        } catch (e) {
          reject(new Error('Invalid JSON from Python: ' + stdout))
        }
      } else {
        reject(new Error(`Python exited ${code}:\n${stderr}`))
      }
    })
  })
})
