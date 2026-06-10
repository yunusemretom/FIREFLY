// Electron Preload Process
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  sendControlCommand: (command) => ipcRenderer.send('control-command', command),
  onTelemetryUpdate: (callback) => ipcRenderer.on('telemetry-update', (_event, value) => callback(value))
});
