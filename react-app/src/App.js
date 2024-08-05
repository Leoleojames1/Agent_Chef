import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import AceEditor from 'react-ace';
import 'ace-builds/src-noconflict/mode-json';
import 'ace-builds/src-noconflict/theme-monokai';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#ff9800',
    },
    secondary: {
      main: '#f44336',
    },
    background: {
      default: '#303030',
      paper: 'rgba(48, 48, 48, 0.8)',
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backdropFilter: 'blur(10px)',
          backgroundColor: 'rgba(48, 48, 48, 0.8)',
        },
      },
    },
  },
});

function App() {
  const [mode, setMode] = useState('');
  const [ollamaModel, setOllamaModel] = useState('');
  const [ingredientFiles, setIngredientFiles] = useState([]);
  const [dishFiles, setDishFiles] = useState([]);
  const [latexFiles, setLatexFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [editorContent, setEditorContent] = useState('');
  const [template, setTemplate] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [output, setOutput] = useState('');
  const [filename, setFilename] = useState('');
  const [templates, setTemplates] = useState({});

  useEffect(() => {
    fetchFiles();
    fetchTemplates();
  }, []);

  const fetchFiles = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/files');
      setIngredientFiles(response.data.ingredient_files);
      setDishFiles(response.data.dish_files);
      setLatexFiles(response.data.latex_files);
    } catch (error) {
      console.error('Error fetching files:', error);
      alert('Error fetching files');
    }
  };

  const fetchTemplates = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/templates');
      setTemplates(response.data);
    } catch (error) {
      console.error('Error fetching templates:', error);
      alert('Error fetching templates');
    }
  };

  const selectFile = async (filename, type) => {
    try {
      const response = await axios.get(`http://localhost:5000/api/file/${type}/${filename}`);
      setSelectedFile(filename);
      setEditorContent(response.data.content);
      setFilename(filename);
    } catch (error) {
      console.error('Error fetching file content:', error);
      alert('Error fetching file content');
    }
  };

  const saveFile = async () => {
    try {
      await axios.post('http://localhost:5000/api/save', {
        filename,
        content: editorContent,
        type: filename.endsWith('.json') ? 'ingredient' : 'dish',
      });
      alert('File saved successfully');
      fetchFiles();
    } catch (error) {
      console.error('Error saving file:', error);
      alert('Error saving file');
    }
  };

  const runAgentChef = async () => {
    try {
      setOutput("Processing...");
      const response = await axios.post('http://localhost:5000/api/run', {
        mode,
        ollamaModel,
        template,
        systemPrompt,
        selectedFile,
      });
      setOutput(response.data.output);
      fetchFiles(); // Refresh file lists after running
    } catch (error) {
      console.error('Error running Agent Chef:', error);
      setOutput(`Error: ${error.response?.data?.error || error.message}`);
    }
  };

  const createTemplate = async () => {
    const templateData = {}; // Collect template data from the user
    const filename = prompt("Enter a name for the new template:");
    if (!filename) return;

    try {
      const response = await axios.post('http://localhost:5000/api/template/create', {
        template_data: templateData,
        filename,
      });
      alert('Template created successfully');
      fetchTemplates(); // Refresh template list
    } catch (error) {
      console.error('Error creating template:', error);
      alert('Error creating template');
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg" sx={{
        backgroundImage: 'linear-gradient(45deg, #ff9800, #f44336, #000)',
        minHeight: '100vh',
        padding: '20px',
      }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ color: '#fff' }}>
          🍲Agent Chef🥩
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 2 }}>
              <Typography variant="h5" gutterBottom>Setup</Typography>
              <Select
                fullWidth
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                displayEmpty
                sx={{ mb: 2 }}
              >
                <MenuItem value="">Select Mode</MenuItem>
                <MenuItem value="custom">Custom</MenuItem>
                <MenuItem value="huggingface">Hugging Face</MenuItem>
                <MenuItem value="latex">LaTeX</MenuItem>
              </Select>
              <TextField
                fullWidth
                label="Ollama Model"
                value={ollamaModel}
                onChange={(e) => setOllamaModel(e.target.value)}
                sx={{ mb: 2 }}
              />
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 2 }}>
              <Typography variant="h5" gutterBottom>Ingredients (JSON)</Typography>
              <List>
                {ingredientFiles.map(file => (
                  <ListItem button key={file} onClick={() => selectFile(file, 'ingredient')}>
                    <ListItemText primary={file} />
                  </ListItem>
                ))}
              </List>
              <Typography variant="h5" gutterBottom>Dishes (Parquet)</Typography>
              <List>
                {dishFiles.map(file => (
                  <ListItem button key={file} onClick={() => selectFile(file, 'dish')}>
                    <ListItemText primary={file} />
                  </ListItem>
                ))}
              </List>
              <Typography variant="h5" gutterBottom>LaTeX Files</Typography>
              <List>
                {latexFiles.map(file => (
                  <ListItem button key={file} onClick={() => selectFile(file, 'latex')}>
                    <ListItemText primary={file} />
                  </ListItem>
                ))}
              </List>
            </Paper>
          </Grid>
          
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2 }}>
              <Typography variant="h5" gutterBottom>Editor</Typography>
              <TextField
                fullWidth
                label="Filename"
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
                sx={{ mb: 2 }}
              />
              <AceEditor
                mode={filename.endsWith('.json') ? 'json' : 'text'}
                theme="monokai"
                onChange={setEditorContent}
                value={editorContent}
                name="editor"
                editorProps={{ $blockScrolling: true }}
                width="100%"
                height="300px"
              />
              <Button variant="contained" onClick={saveFile} sx={{ mt: 2 }}>
                Save File
              </Button>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 2 }}>
              <Typography variant="h5" gutterBottom>Template</Typography>
              <Select
                fullWidth
                value={template}
                onChange={(e) => setTemplate(e.target.value)}
                displayEmpty
                sx={{ mb: 2 }}
              >
                <MenuItem value="">Select Template</MenuItem>
                {Object.keys(templates).map((templateName) => (
                  <MenuItem key={templateName} value={templateName}>
                    {templateName}
                  </MenuItem>
                ))}
                <MenuItem value="custom">Custom</MenuItem>
              </Select>
              <Button variant="contained" onClick={createTemplate} sx={{ mt: 2 }}>
                Create Template
              </Button>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 2 }}>
              <Typography variant="h5" gutterBottom>System Prompt</Typography>
              <TextField
                fullWidth
                multiline
                rows={4}
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
              />
            </Paper>
          </Grid>
          
          <Grid item xs={12}>
            <Button variant="contained" onClick={runAgentChef} fullWidth>
              Run Agent Chef
            </Button>
          </Grid>
          
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2 }}>
              <Typography variant="h5" gutterBottom>Output</Typography>
              <pre>{output}</pre>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </ThemeProvider>
  );
}

export default App;