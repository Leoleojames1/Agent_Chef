import React, { useState, useEffect, useRef } from 'react';
import './App.css';
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
import Checkbox from '@mui/material/Checkbox';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormGroup from '@mui/material/FormGroup';
import Switch from '@mui/material/Switch';
import AceEditor from 'react-ace';
import 'ace-builds/src-noconflict/mode-json';
import 'ace-builds/src-noconflict/mode-python';
import 'ace-builds/src-noconflict/mode-javascript';
import 'ace-builds/src-noconflict/mode-html';
import 'ace-builds/src-noconflict/mode-css';
import 'ace-builds/src-noconflict/mode-text';
import 'ace-builds/src-noconflict/theme-monokai';
import { styled } from '@mui/system';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TablePagination from '@mui/material/TablePagination';

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
      default: 'transparent',
      paper: 'rgba(48, 48, 48, 0.8)',
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backdropFilter: 'blur(10px)',
          backgroundColor: 'rgba(18, 18, 18, 0.8)',
        },
      },
    },
  },
});

const PerlinNoiseBackground = styled('div')({
  position: 'fixed',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%',
  zIndex: -1,
  background: `
    linear-gradient(to bottom right, 
      rgba(0, 0, 50, 0.8), 
      rgba(0, 0, 0, 0.8)
    ),
    url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.15' numOctaves='3' stitchTiles='stitch'/%3E%3Canimate attributeName='baseFrequency' values='0.15; 0.2; 0.25; 0.2' dur='40s' repeatCount='indefinite'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")
  `,
  backgroundBlendMode: 'overlay',
  animation: 'noise 40s steps(4) infinite',
});

const MainContent = styled(Box)({
  flexGrow: 1,
  padding: theme.spacing(3),
  transition: theme.transitions.create('margin', {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
});

const ActionButton = styled(Button)(({ theme }) => ({
  margin: theme.spacing(1),
}));

function ParquetViewer({ data, columns, totalRows }) {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  return (
    <>
      <TableContainer component={Paper} sx={{ maxHeight: '50vh' }}>
        <Table stickyHeader aria-label="parquet data table">
          <TableHead>
            <TableRow>
              {columns.map((column) => (
                <TableCell key={column}>{column}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {data.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((row, index) => (
              <TableRow key={index}>
                {columns.map((column) => (
                  <TableCell key={column}>{row[column]}</TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[10, 25, 100]}
        component="div"
        count={totalRows}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
    </>
  );
}

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
  const [seeds, setSeeds] = useState([]);
  const [selectedSeed, setSelectedSeed] = useState(null);
  const [seedMessage, setSeedMessage] = useState('No seeds available');  // Add this line
  const [parquetData, setParquetData] = useState(null);
  const [parquetColumns, setParquetColumns] = useState([]);
  const [totalRows, setTotalRows] = useState(0);
  const [fileType, setFileType] = useState(null);  // Add this line
  const [ollamaModels, setOllamaModels] = useState([]);
  const [selectedOllamaModel, setSelectedOllamaModel] = useState('');
  const [customSeedFilename, setCustomSeedFilename] = useState('');
  const [customSeedText, setCustomSeedText] = useState('');
  const [txtContent, setTxtContent] = useState('');
  const [txtFilename, setTxtFilename] = useState('');
  const [proceduralParquet, setProceduralParquet] = useState(null);
  const [selectedLatexFile, setSelectedLatexFile] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [editorMode, setEditorMode] = useState('text');
  const [builtSeed, setBuiltSeed] = useState(null);
  const [allFiles, setAllFiles] = useState([]);
  const [selectedFileType, setSelectedFileType] = useState(null);
  const [formattingMode, setFormattingMode] = useState('manual');
  const aceEditorRef = useRef(null);
  const [sampleRate, setSampleRate] = useState(100); // Default to 100%
  const [totalSamples, setTotalSamples] = useState(0);
  const [selectedSample, setSelectedSample] = useState(null);
  const [numParaphrases, setNumParaphrases] = useState(5);
  const [paraphrases, setParaphrases] = useState([]);
  const [paraphrasesPerSample, setParaphrasesPerSample] = useState(5);
  const [staticColumns, setStaticColumns] = useState([]);
  const [availableColumns, setAvailableColumns] = useState([]);
  const [columnTypes, setColumnTypes] = useState({});
  const [useAllSamples, setUseAllSamples] = useState(true);

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
    fetchSeeds();
    fetchOllamaModels();
  }, []);

useEffect(() => {
  if (availableColumns.length > 0) {
    const initialColumnTypes = availableColumns.reduce((acc, column) => {
      if (column === 'command_description') acc[column] = 'static';
      else if (column === 'command') acc[column] = 'reference';
      else acc[column] = 'dynamic';
      return acc;
    }, {});
    setColumnTypes(initialColumnTypes);
    console.log("Initialized column types:", initialColumnTypes);
  }
}, [availableColumns]);

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

  const handleColumnTypeToggle = (column) => {
    setColumnTypes(prevTypes => ({
      ...prevTypes,
      [column]: prevTypes[column] === 'static' ? 'dynamic' : 'static'
    }));
  };
  
  const handleColumnTypeChange = (column, newType) => {
    setColumnTypes(prevTypes => ({
      ...prevTypes,
      [column]: newType
    }));
  };

  const fetchOllamaModels = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/ollama-models');
      setOllamaModels(response.data.models);
    } catch (error) {
      console.error('Error fetching Ollama models:', error);
      setError('Error fetching Ollama models');
    }
  };

  const handleFileSelection = (file) => {
    setSelectedFiles(prevSelected => {
      if (prevSelected.some(f => f.name === file.name && f.type === file.type)) {
        return prevSelected.filter(f => f.name !== file.name || f.type !== file.type);
      } else {
        return [...prevSelected, file];
      }
    });
  };

  const saveTxtFile = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/save', {
        filename: txtFilename,
        content: txtContent,
        type: 'ingredient',
      });
      if (response.data.success) {
        alert('Text file saved successfully');
        fetchFiles();
      } else {
        setError(`Error saving text file: ${response.data.message}`);
      }
    } catch (error) {
      setError(`Error saving text file: ${error.response?.data?.message || error.message}`);
    }
  };

  const buildProceduralParquet = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/convert_to_parquet', {
        content: txtContent,
        template: selectedTemplate,
        filename: txtFilename,
      });
      setProceduralParquet(response.data.parquet_file);
      setOutput(`Procedural parquet created: ${response.data.parquet_file}`);
      fetchFiles();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const generateSyntheticParquet = async () => {
    try {
      if (!proceduralParquet) {
        throw new Error("No procedural parquet file selected.");
      }
      
      const response = await axios.post('http://localhost:5000/api/generate_synthetic', {
        seed_parquet: proceduralParquet,
        num_samples: numSamples,
        ollama_model: selectedOllamaModel,
        system_prompt: systemPrompt,
      });
      
      setOutput(`Synthetic parquet created: ${response.data.filename}`);
      fetchFiles();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const useLatexFile = async (filename) => {
    try {
      const response = await axios.get(`http://localhost:5000/api/file/latex/${filename}`);
      setTxtContent(response.data.content);
      setTxtFilename(filename.replace('.tex', '.txt'));
      setSelectedLatexFile(filename);
    } catch (error) {
      setError(`Error loading LaTeX file: ${error.response?.data?.message || error.message}`);
    }
  };

  const fetchSeeds = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/seeds');
      if (response.data.seeds && response.data.seeds.length > 0) {
        setSeeds(response.data.seeds);
        setSeedMessage('');  // Clear the message if seeds are available
      } else {
        setSeeds([]);
        setSeedMessage('No seeds available');
      }
    } catch (error) {
      console.error('Error fetching seeds:', error);
      setError('Error fetching seeds');
      setSeedMessage('Error fetching seeds');
    }
  };

  const buildSeed = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/convert_to_json_parquet', {
        filename,
      });
      setBuiltSeed(response.data.parquet_file);
      setOutput(`Seed parquet created: ${response.data.parquet_file}`);
      fetchFiles();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const combineSelectedFiles = async () => {
    if (selectedFiles.length < 2) {
      setError("Please select at least two files to combine.");
      return;
    }
    
    const fileTypes = new Set(selectedFiles.map(file => file.name.split('.').pop()));
    if (fileTypes.size > 1) {
      setError("Can only combine files of the same type.");
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/api/combine_files', {
        files: selectedFiles,
      });
      setOutput(`Combined file created: ${response.data.combined_file}`);
      fetchFiles();
      setSelectedFiles([]);  // Clear selection after combining
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const fetchFiles = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/files');
      setIngredientFiles(response.data.ingredient_files);
      setDishFiles(response.data.dish_files);
      setLatexFiles(response.data.latex_files);
      setAllFiles([
        ...response.data.ingredient_files.map(file => ({ name: file, type: 'ingredient' })),
        ...response.data.dish_files.map(file => ({ name: file, type: 'dish' })),
        ...response.data.latex_files.map(file => ({ name: file, type: 'latex' }))
      ]);
    } catch (error) {
      console.error('Error fetching files:', error);
      setError('Error fetching files');
    }
  };

  const selectSample = async (filename) => {
    try {
      const response = await axios.get(`http://localhost:5000/api/file/ingredient/${filename}`);
      setSelectedSample(response.data.content[0]);  // Assuming the first row is selected
    } catch (error) {
      console.error('Error fetching sample content:', error);
      setError('Error fetching sample content');
    }
  };

  const selectSeed = async (filename) => {
    try {
      const response = await axios.get(`http://localhost:5000/api/file/ingredient/${filename}`);
      setSelectedSeed(filename);
      setEditorContent(JSON.stringify(response.data.content, null, 2));
    } catch (error) {
      console.error('Error fetching seed content:', error);
      setError('Error fetching seed content');
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
    setSelectedFile(filename);
    setSelectedFileType(type);
    setFilename(filename);
    
    try {
      const response = await axios.get(`http://localhost:5000/api/file/${type}/${filename}`);
      
    if (filename.endsWith('.parquet')) {
      setFileType('parquet');
      setParquetData(response.data.content);
      setParquetColumns(response.data.columns);
      setTotalRows(response.data.total_rows);
      setAvailableColumns(response.data.columns);
      setEditorContent('');
    } else {
        setFileType(filename.split('.').pop());
        if (typeof response.data.content === 'string') {
          setEditorContent(response.data.content);
        } else {
          setEditorContent(JSON.stringify(response.data.content, null, 2));
        }
        setParquetData(null);
        setParquetColumns([]);
        setTotalRows(0);
      }
      
      // Set editor mode based on file extension
      if (filename.endsWith('.json')) {
        setEditorMode('json');
      } else if (filename.endsWith('.py')) {
        setEditorMode('python');
      } else if (filename.endsWith('.js')) {
        setEditorMode('javascript');
      } else if (filename.endsWith('.html')) {
        setEditorMode('html');
      } else if (filename.endsWith('.css')) {
        setEditorMode('css');
      } else {
        setEditorMode('text');
      }
    } catch (error) {
      console.error('Error fetching file content:', error);
      setError('Error fetching file content');
    }
  };

  const handleColumnToggle = (column) => {
    setStaticColumns(prev => 
      prev.includes(column) 
        ? prev.filter(c => c !== column) 
        : [...prev, column]
    );
  };

  const insertSymbol = (symbol) => {
    const editor = aceEditorRef.current.editor;
    editor.insert(`$("${symbol}")`);
    editor.focus();
  };

  const parseDataset = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/parse_dataset', {
        content: txtContent,
        template: selectedTemplate,
        mode: formattingMode,
      });
      if (response.data.success) {
        setOutput(JSON.stringify(response.data.result, null, 2));
        fetchFiles();
      } else {
        setError(`Error parsing dataset: ${response.data.message}`);
      }
    } catch (error) {
      setError(`Error parsing dataset: ${error.response?.data?.message || error.message}`);
    }
  };
  
  const saveFile = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/save', {
        filename,
        content: editorContent,
        type: 'ingredient',
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
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

  const convertToJson = async () => {
    try {
      if (!selectedFile || !selectedTemplate) {
        throw new Error("Please select a file and a template");
      }
  
      let content = editorContent;
      if (fileType === 'txt') {
        // For TXT files, we need to send the content as is
        const response = await axios.get(`http://localhost:5000/api/file/${selectedFileType}/${selectedFile}`);
        content = response.data.content;
      }
  
      const response = await axios.post('http://localhost:5000/api/convert_to_json', {
        filename: selectedFile,
        content: content,
        template: selectedTemplate,
        fileType: selectedFileType
      });
  
      setOutput(`JSON file created: ${response.data.json_file}, Parquet file created: ${response.data.parquet_file}`);
      fetchFiles();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const convertJsonToParquetSeed = async () => {
    try {
      if (!selectedFile || !selectedFile.endsWith('.json')) {
        throw new Error("Please select a JSON file to convert to parquet");
      }

      const response = await axios.post('http://localhost:5000/api/convert_to_json_parquet', {
        filename: selectedFile,
      });

      setBuiltSeed(response.data.parquet_file);
      setOutput(`Seed parquet created: ${response.data.parquet_file}`);
      fetchFiles();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
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

  const saveCustomSeed = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/save', {
        filename: customSeedFilename,
        content: customSeedText,
        type: 'ingredient',
      });
      if (response.data.success) {
        alert('Custom seed file saved successfully');
        fetchFiles();
        fetchSeeds();
      } else {
        setError(`Error saving custom seed file: ${response.data.message}`);
      }
    } catch (error) {
      console.error('Error saving custom seed file:', error.response?.data || error.message);
      setError(`Error saving custom seed file: ${error.response?.data?.message || error.message}`);
    }
  };

  const useCustomSeedForSynthetic = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/convert_to_json_parquet', {
        content: customSeedText,
        template: selectedTemplate,
        filename: customSeedFilename,
      });
      setSelectedSeed(response.data.parquet_file);
      setOutput(`Seed parquet created: ${response.data.parquet_file}`);
      fetchFiles();
      fetchSeeds();
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };
  
  const generateParaphrases = async () => {
    if (!selectedSample) {
      setError("Please select a sample first");
      return;
    }
  
    try {
      const response = await axios.post('http://localhost:5000/api/generate_paraphrases', {
        sample: selectedSample,
        num_paraphrases: numParaphrases,
        ollama_model: selectedOllamaModel
      });
      
      setParaphrases(response.data.paraphrases);
      setOutput(`Generated ${response.data.paraphrases.length} paraphrases`);
    } catch (error) {
      setError(error.response?.data?.error || error.message);
    }
  };

  const runAgentChef = async () => {
    try {
      setError(null);
      setOutput("Processing...");
      
      if (!selectedFile) {
        throw new Error("Please select a file for generating synthetic data.");
      }
      
      if (!selectedOllamaModel) {
        throw new Error("No Ollama model selected. Please select a model.");
      }
      
      const dataToSend = {
        mode: 'custom',
        ollamaModel: selectedOllamaModel,
        seedFile: selectedFile,
        systemPrompt: systemPrompt,
        sampleRate: sampleRate,
        paraphrasesPerSample: paraphrasesPerSample,
        columnTypes: columnTypes,  // This should now contain the correct column types
        useAllSamples: useAllSamples
      };
  
      console.log("Sending data to server:", JSON.stringify(dataToSend, null, 2));
  
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
      <PerlinNoiseBackground />
      <Box sx={{ display: 'flex' }}>
        <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
          <Toolbar>
            <Typography variant="h4" noWrap component="div">
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
                <Typography variant="h6">Ingredients</Typography>
              </ListItem>
              {allFiles.filter(file => file.type === 'ingredient').map(file => (
                <ListItem 
                  key={`${file.type}-${file.name}`} 
                  disablePadding 
                  sx={{
                    backgroundColor: selectedFiles.some(f => f.name === file.name && f.type === file.type) 
                      ? 'rgba(25, 118, 210, 0.12)' 
                      : 'transparent',
                  }}
                >
                  <ListItemButton 
                    onClick={() => handleFileSelection(file)}
                    onDoubleClick={() => selectFile(file.name, file.type)}
                  >
                    <ListItemText primary={file.name} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
            <Divider />
            <List>
              <ListItem>
                <Typography variant="h6">Dishes</Typography>
              </ListItem>
              {allFiles.filter(file => file.type === 'dish').map(file => (
                <ListItem 
                  key={`${file.type}-${file.name}`} 
                  disablePadding 
                  sx={{
                    backgroundColor: selectedFiles.some(f => f.name === file.name && f.type === file.type) 
                      ? 'rgba(25, 118, 210, 0.12)' 
                      : 'transparent',
                  }}
                >
                  <ListItemButton 
                    onClick={() => handleFileSelection(file)}
                    onDoubleClick={() => selectFile(file.name, file.type)}
                  >
                    <ListItemText primary={file.name} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
            {latexFiles.length > 0 && (
              <>
                <Divider />
                <List>
                  <ListItem>
                    <Typography variant="h6">LaTeX Files</Typography>
                  </ListItem>
                  {allFiles.filter(file => file.type === 'latex').map(file => (
                    <ListItem 
                      key={`${file.type}-${file.name}`} 
                      disablePadding 
                      sx={{
                        backgroundColor: selectedFiles.some(f => f.name === file.name && f.type === file.type) 
                          ? 'rgba(25, 118, 210, 0.12)' 
                          : 'transparent',
                      }}
                    >
                      <ListItemButton 
                        onClick={() => handleFileSelection(file)}
                        onDoubleClick={() => selectFile(file.name, file.type)}
                      >
                        <ListItemText primary={file.name} />
                      </ListItemButton>
                    </ListItem>
                  ))}
                </List>
              </>
            )}
        </Box>
      </Drawer>
      <MainContent>
        <Toolbar />
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
              <Typography variant="h5" gutterBottom>Setup</Typography>
              <Select
                fullWidth
                value={selectedOllamaModel}
                onChange={(e) => setSelectedOllamaModel(e.target.value)}
                displayEmpty
                sx={{ mb: 2 }}
              >
                <MenuItem value="">Select Ollama Model</MenuItem>
                {ollamaModels.map((model) => (
                  <MenuItem key={model} value={model}>{model}</MenuItem>
                ))}
              </Select>
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
                multiline
                rows={4}
                label="System Prompt"
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Random Sample Rate (%)"
                type="number"
                value={sampleRate}
                onChange={(e) => setSampleRate(Math.max(0, Math.min(100, parseInt(e.target.value) || 0)))}
                inputProps={{ min: 0, max: 100 }}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Paraphrases per Sample"
                type="number"
                value={paraphrasesPerSample}
                onChange={(e) => setParaphrasesPerSample(Math.max(1, parseInt(e.target.value) || 1))}
                inputProps={{ min: 1 }}
                sx={{ mb: 2 }}
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={useAllSamples}
                    onChange={(e) => setUseAllSamples(e.target.checked)}
                  />
                }
                label="Use All Samples (Negates Random Sample Rate)"
                
              />
            </Paper>
          </Grid>
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
              <Typography variant="h5" gutterBottom>Editor / Viewer</Typography>
              <Typography variant="body2" sx={{ mb: 2 }}>
                Note: You can include multi-line content, such as Python code, within a single $("data/code") group split across multiple lines. Run the Parse Dataset button to have Agent Chef attempt to generate your $("data") formatting for the provided txt file and based on the selected format or format the data by hand.
              </Typography>
              <Box sx={{ mb: 2 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={formattingMode === 'automatic'}
                      onChange={(e) => setFormattingMode(e.target.checked ? 'automatic' : 'manual')}
                    />
                  }
                  label="Automatic Formatting"
                />
              </Box>
              <TextField
                fullWidth
                label="Filename"
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
                sx={{ mb: 2 }}
              />
              {fileType === 'parquet' ? (
                <ParquetViewer data={parquetData} columns={parquetColumns} totalRows={totalRows} />
              ) : (
                <AceEditor
                  ref={aceEditorRef}
                  mode={editorMode}
                  theme="monokai"
                  onChange={setEditorContent}
                  value={editorContent}
                  name="editor"
                  editorProps={{ $blockScrolling: true }}
                  width="100%"
                  height="400px"
                  setOptions={{
                    useWorker: false,
                    showLineNumbers: true,
                    tabSize: 2,
                  }}
                />
              )}
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
                <Button variant="contained" onClick={saveFile}>
                  Save as TXT
                </Button>
                <Button 
                  variant="contained" 
                  onClick={convertToJson}
                  disabled={!selectedTemplate || !selectedFile || !(fileType === 'txt' || fileType === 'tex' || fileType === 'json')}
                >
                  Convert to JSON & Parquet Seeds
                </Button>
                <Button 
                  variant="contained" 
                  onClick={combineSelectedFiles}
                  disabled={selectedFiles.length < 2}
                >
                  Combine Selected Files
                </Button>
                <Button variant="contained" onClick={parseDataset}>
                  Parse Dataset
                </Button>
              </Box>
              {selectedFiles.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle1">Selected Files:</Typography>
                  <ul>
                    {selectedFiles.map(file => (
                      <li key={`${file.type}-${file.name}`}>{file.name} ({file.type})</li>
                    ))}
                  </ul>
                </Box>
              )}
            </Paper>
            </Grid>
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
              <Typography variant="h5" gutterBottom>Column Selection</Typography>
              <Typography variant="body2" gutterBottom>
                Select the type for each column:
                - Static: Copied directly from the original
                - Reference: Used for augmenting dynamic columns
                - Dynamic: Paraphrased/generated
              </Typography>
              {availableColumns.map((column) => (
                <Box key={column} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography sx={{ minWidth: 150 }}>{column}:</Typography>
                  <Select
                    value={columnTypes[column] || 'dynamic'}
                    onChange={(e) => handleColumnTypeChange(column, e.target.value)}
                    sx={{ minWidth: 150 }}
                  >
                    <MenuItem value="static">Static</MenuItem>
                    <MenuItem value="reference">Reference</MenuItem>
                    <MenuItem value="dynamic">Dynamic</MenuItem>
                  </Select>
                </Box>
              ))}
            </Paper>
          </Grid>
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
              <Typography variant="h5" gutterBottom>Synthetic Data Generation</Typography>
              <Button 
                fullWidth
                variant="contained" 
                onClick={runAgentChef}
                disabled={!selectedOllamaModel || !selectedFile || !(fileType === 'parquet' || fileType === 'txt' || fileType === 'json')}
                sx={{ mb: 2 }}
              >
                Generate Synthetic Data
              </Button>
            </Paper>
          </Grid>
          {error && (
            <Grid item xs={12}>
              <Paper elevation={3} sx={{ p: 2, mb: 2, bgcolor: 'error.main' }}>
                <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>Error</Typography>
                <Typography sx={{ color: 'white' }}>{error}</Typography>
              </Paper>
            </Grid>
          )}
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2 }}>
              <Typography variant="h5" gutterBottom>Output</Typography>
              <pre>{output}</pre>
            </Paper>
          </Grid>
        </Grid>
      </MainContent>
    </Box>
  </ThemeProvider>
);
}

export default App;