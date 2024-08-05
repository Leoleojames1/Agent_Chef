import React, { useState, useEffect } from 'react';
import axios from 'axios';
import AceEditor from 'react-ace';
import 'ace-builds/src-noconflict/mode-json';
import 'ace-builds/src-noconflict/theme-monokai';

function App() {
  const [mode, setMode] = useState('');
  const [ollamaModel, setOllamaModel] = useState('');
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [editorContent, setEditorContent] = useState('');
  const [template, setTemplate] = useState({});
  const [systemPrompt, setSystemPrompt] = useState('');
  const [output, setOutput] = useState('');

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/files');
      setFiles(response.data.files);
    } catch (error) {
      console.error('Error fetching files:', error);
    }
  };

  const selectFile = async (filename) => {
    try {
      const response = await axios.get(`http://localhost:5000/api/file/${filename}`);
      setSelectedFile(filename);
      setEditorContent(response.data.content);
    } catch (error) {
      console.error('Error fetching file content:', error);
    }
  };

  const saveFile = async () => {
    try {
      await axios.post('http://localhost:5000/api/save', {
        filename: selectedFile,
        content: editorContent
      });
      alert('File saved successfully!');
    } catch (error) {
      console.error('Error saving file:', error);
    }
  };

  const runAgentChef = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/run', {
        mode,
        ollamaModel,
        template,
        systemPrompt
      });
      setOutput(response.data.output);
    } catch (error) {
      console.error('Error running Agent Chef:', error);
    }
  };

  return (
    <div className="app">
      <h1>Agent Chef Web Interface</h1>
      
      <div className="setup">
        <h2>Setup</h2>
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="">Select Mode</option>
          <option value="custom">Custom</option>
          <option value="huggingface">Hugging Face</option>
          <option value="json">JSON</option>
          <option value="build">Build JSON</option>
        </select>
        <input 
          type="text" 
          value={ollamaModel} 
          onChange={(e) => setOllamaModel(e.target.value)} 
          placeholder="Ollama Model"
        />
      </div>

      <div className="file-list">
        <h2>Files</h2>
        <ul>
          {files.map(file => (
            <li key={file} onClick={() => selectFile(file)}>{file}</li>
          ))}
        </ul>
      </div>

      <div className="editor">
        <h2>Editor</h2>
        <AceEditor
          mode="json"
          theme="monokai"
          onChange={setEditorContent}
          value={editorContent}
          name="editor"
          editorProps={{ $blockScrolling: true }}
        />
        <button onClick={saveFile}>Save</button>
      </div>

      <div className="template">
        <h2>Template</h2>
        <textarea 
          value={JSON.stringify(template, null, 2)} 
          onChange={(e) => setTemplate(JSON.parse(e.target.value))}
        />
      </div>

      <div className="system-prompt">
        <h2>System Prompt</h2>
        <textarea 
          value={systemPrompt} 
          onChange={(e) => setSystemPrompt(e.target.value)}
        />
      </div>

      <button onClick={runAgentChef}>Run Agent Chef</button>

      <div className="output">
        <h2>Output</h2>
        <pre>{output}</pre>
      </div>
    </div>
  );
}

export default App;