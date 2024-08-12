import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Drawer from '@mui/material/Drawer';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import AceEditor from 'react-ace';
import 'ace-builds/src-noconflict/mode-json';
import 'ace-builds/src-noconflict/theme-monokai';

const drawerWidth = 240;

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#4fc3f7',
    },
    secondary: {
      main: '#f48fb1',
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
  const [systemPrompt, setSystemPrompt] = useState('');
  const [output, setOutput] = useState('');
  const [filename, setFilename] = useState('');
  const [templates, setTemplates] = useState({});
  const [error, setError] = useState(null);
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [syntheticTechnique, setSyntheticTechnique] = useState('');
  const [newTemplateName, setNewTemplateName] = useState('');
  const [newTemplateColumns, setNewTemplateColumns] = useState('');
  const [numSamples, setNumSamples] = useState(100);
  const [newTemplateFields, setNewTemplateFields] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/templates');
        console.log("Fetched templates:", response.data);
        setTemplates(response.data || {});
      } catch (error) {
        console.error('Error fetching templates:', error);
        setError('Error fetching templates: ' + error.message);
      }
    };

    fetchData();
  }, []);

  useEffect(() => {
    fetchFiles();
    fetchTemplates();
  }, []);


  if (error) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box sx={{ p: 3 }}>
          <Typography color="error">{error}</Typography>
        </Box>
      </ThemeProvider>
    );
  }

  const fetchFiles = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/files');
      setIngredientFiles(response.data.ingredient_files);
      setDishFiles(response.data.dish_files);
      setLatexFiles(response.data.latex_files);
    } catch (error) {
      console.error('Error fetching files:', error);
      setError('Error fetching files');
    }
  };

  const fetchTemplates = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/templates');
      console.log("Fetched templates:", response.data);
      setTemplates(response.data || {});
    } catch (error) {
      console.error('Error fetching templates:', error);
      setError('Error fetching templates');
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
      setError('Error fetching file content');
    }
  };

  const saveFile = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/save', {
        filename,
        content: editorContent,
        type: 'ingredient',
      });
      if (response.data.success) {
        alert('File saved successfully');
        fetchFiles();
      } else {
        setError(`Error saving file: ${response.data.message}`);
      }
    } catch (error) {
      console.error('Error saving file:', error.response?.data || error.message);
      setError(`Error saving file: ${error.response?.data?.message || error.message}`);
    }
  };

  const convertToJsonParquet = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/convert_to_json_parquet', {
        content: editorContent,
        template: selectedTemplate,
        filename,
      });
      setOutput(`JSON file: ${response.data.json_file}, Parquet file: ${response.data.parquet_file}`);
      fetchFiles();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const runAgentChef = async () => {
    try {
      setError(null);
      setOutput("Processing...");
      
      if (!selectedFile) {
        throw new Error("No seed parquet file selected");
      }
      
      const dataToSend = {
        mode,
        ollamaModel,
        seedParquet: selectedFile,
        syntheticTechnique,
        template: selectedTemplate,
        systemPrompt,
        numSamples: parseInt(numSamples, 10)
      };

      console.log("Sending data to server:", dataToSend);

      const response = await axios.post('http://localhost:5000/api/run', dataToSend);
      
      if (response.data.error) {
        setError(response.data.error);
        setOutput('');
      } else {
        setOutput(JSON.stringify(response.data, null, 2));
        fetchFiles();
      }
    } catch (error) {
      setError(error.response?.data?.error || error.message);
      setOutput('');
    }
  };

  const addTemplate = async () => {
    try {
      await axios.post('http://localhost:5000/api/templates', {
        name: newTemplateName,
        fields: newTemplateFields.split(',').map(field => field.trim())
      });
      fetchTemplates();
      setNewTemplateName('');
      setNewTemplateFields('');
    } catch (error) {
      console.error('Error adding template:', error);
      setError(`Error adding template: ${error.response?.data?.message || error.message}`);
    }
  };

  const renderTemplateOptions = () => {
    if (!templates || Object.keys(templates).length === 0) {
      return <MenuItem value="">No templates available</MenuItem>;
    }
    return Object.entries(templates).map(([name, fields]) => (
      <MenuItem key={name} value={name}>
        {name}: {Array.isArray(fields) ? fields.join(', ') : 'Invalid template'}
      </MenuItem>
    ));
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex' }}>
        <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
          <Toolbar>
            <Typography variant="h6" noWrap component="div">
              🍲Agent Chef🥩
            </Typography>
          </Toolbar>
        </AppBar>
        <Drawer
          variant="permanent"
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: 'auto' }}>
            <List>
              <ListItem>
                <Typography variant="h6">Ingredients (JSON/Parquet)</Typography>
              </ListItem>
              {ingredientFiles.map(file => (
                <ListItem key={file} disablePadding>
                  <ListItemButton onClick={() => selectFile(file, 'ingredient')}>
                    <ListItemText primary={file} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
            <Divider />
            <List>
              <ListItem>
                <Typography variant="h6">Dishes (Synthetic Data)</Typography>
              </ListItem>
              {dishFiles.map(file => (
                <ListItem key={file} disablePadding>
                  <ListItemButton onClick={() => selectFile(file, 'dish')}>
                    <ListItemText primary={file} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
            <Divider />
            <List>
              <ListItem>
                <Typography variant="h6">LaTeX Files</Typography>
              </ListItem>
              {latexFiles.map(file => (
                <ListItem key={file} disablePadding>
                  <ListItemButton onClick={() => selectFile(file, 'latex')}>
                    <ListItemText primary={file} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </Box>
        </Drawer>
        <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
          <Toolbar />
          <Grid container spacing={2}>
            <Grid item xs={12} md={8}>
              <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
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
                  <MenuItem value="parquet">Parquet</MenuItem>
                </Select>
                <TextField
                  fullWidth
                  label="Ollama Model"
                  value={ollamaModel}
                  onChange={(e) => setOllamaModel(e.target.value)}
                  sx={{ mb: 2 }}
                />
                  <Select
                    fullWidth
                    value={selectedTemplate}
                    onChange={(e) => setSelectedTemplate(e.target.value)}
                    displayEmpty
                    sx={{ mb: 2 }}
                  >
                    <MenuItem value="">Select Template</MenuItem>
                    {renderTemplateOptions()}
                  </Select>
                  <TextField
                    fullWidth
                    label="New Template Name"
                    value={newTemplateName}
                    onChange={(e) => setNewTemplateName(e.target.value)}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="New Template Fields (comma-separated)"
                    value={newTemplateFields}
                    onChange={(e) => setNewTemplateFields(e.target.value)}
                    sx={{ mb: 2 }}
                  />
                  <Button variant="contained" onClick={addTemplate} sx={{ mb: 2 }}>
                    Add New Template
                  </Button>
              </Paper>
              <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
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
                  height="400px"
                />
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper elevation={3} sx={{ p: 2, mb: 2, position: 'sticky', top: '88px' }}>
                <Typography variant="h5" gutterBottom>Actions</Typography>
                <Button 
                  fullWidth
                  variant="contained" 
                  onClick={saveFile} 
                  sx={{ mb: 2 }}
                >
                  Save File
                </Button>
                <Button 
                  fullWidth
                  variant="contained" 
                  onClick={convertToJsonParquet} 
                  sx={{ mb: 2 }}
                >
                  Convert to JSON/Parquet
                </Button>
                <Button 
                  fullWidth
                  variant="contained" 
                  onClick={runAgentChef} 
                  sx={{ mb: 2 }}
                >
                  Run Agent Chef / Generate Synthetic Data
                </Button>
                <TextField
                  fullWidth
                  label="Number of Samples"
                  type="number"
                  value={numSamples}
                  onChange={(e) => setNumSamples(e.target.value)}
                  sx={{ mb: 2 }}
                />
                {mode === 'parquet' && (
                  <Select
                    fullWidth
                    value={syntheticTechnique}
                    onChange={(e) => setSyntheticTechnique(e.target.value)}
                    displayEmpty
                    sx={{ mb: 2 }}
                  >
                    <MenuItem value="">Select Synthetic Technique</MenuItem>
                    <MenuItem value="combine">Combine Parquets</MenuItem>
                    <MenuItem value="augment">Augment Data</MenuItem>
                  </Select>
                )}
                <TextField
                  fullWidth
                  multiline
                  rows={4}
                  label="System Prompt"
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  sx={{ mb: 2 }}
                />
              </Paper>
            </Grid>
          </Grid>
          {error && (
            <Paper elevation={3} sx={{ p: 2, mb: 2, bgcolor: 'error.main' }}>
              <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>Error</Typography>
              <Typography sx={{ color: 'white' }}>{error}</Typography>
            </Paper>
          )}
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h5" gutterBottom>Output</Typography>
            <pre>{output}</pre>
          </Paper>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;