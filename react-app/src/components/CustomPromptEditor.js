import React, { useState, useEffect } from 'react';
import { TextField, Button, Typography, Box, Accordion, AccordionSummary, AccordionDetails, Grid, Select, MenuItem } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import axios from 'axios';

const defaultPrompts = {
  system: `You are helpful function calling dataset construction assistant, please help me build this function description questioning command calling dataset for Ollama Agent Roll Cage. You will be given user requests to paraphrase and generate alternative phrasing questions for. If the sentence is a question, make sure the paraphrased form retains the original questions meaning, DO NOT ANSWER WHAT THE QUESTION ASKS, REPHRASE THE QUESTION ASKED. maintain all of the details of the description given.`,
  paraphrase: {
    system: `You are a dataset paraphrasing assistant. Your task is to maintain all of the details of the description given maintaining its original meaning and incorporating the provided reference values. Do not add any explanatory text or meta-information.`,
    user: `Original text: {text}
Reference values: {reference_values}

Please maintain all of the details of the description given, maintaining its core meaning and incorporating the reference values where appropriate. Ensure the paraphrased text is coherent and contextually relevant. Provide only the paraphrased text without any additional explanations or formatting.
Please Do maintain the structure of the sentence, if the sentence is a question the generated paraphrase should be a similarly asked question with the question mark, if the sentence is a statement, the paraphrase maintain its meaning should be a similary stated statement with a period. DO NOT COPY OUTPUT.`
  },
  verify: {
    system: `You are a verification assistant for Agent Chef a dataset constructor tool. Your task is to ensure that the paraphrased content maintains the original meaning, format (question or statement), and incorporates the reference values correctly. If the paraphrase is accurate, return it as-is. If not, provide a corrected version.`,
    user: `Original: {original}
Paraphrased: {paraphrased}
Reference values: {reference}
Is question: {is_question}

Verify that the paraphrased content maintains the original meaning, format (question or statement), and correctly incorporates the reference values. If it does, return the paraphrased content. If not, provide a corrected version that accurately reflects the original meaning, format, and includes the reference values.
Do not include create any explanatory text or meta-information in your response, instead just utilize the existing meaning.`
  },
  dynamicColumns: {
    input: {
      system: `You are an AI assistant specializing paraphrasing reference data and generating input question data. Your task is to maintain all of the details of the description given maintaining its original meaning and incorporating the provided reference values. Do not add any explanatory text or meta-information. You will keep all questions as questions and just re-ask them.`,
      user: `Given the context and reference values, generate a suitable input question rephrase response:`
    },
    output: {
      system: `You are an AI assistant specializing paraphrasing reference data and generating output statement data. Your task is to maintain all of the details of the description given maintaining its original meaning and incorporating the provided reference values. Do not add any explanatory text or meta-information. You will keep all statements as statements and just re-state them.`,
      user: `Given the context and reference values, generate a suitable output statement rephrase response:`
    }
  }
};

const CustomPromptEditor = ({ columnTypes, onSavePrompts, initialCustomPrompts = {} }) => {
  const [customPrompts, setCustomPrompts] = useState(() => ({
    ...defaultPrompts,
    ...initialCustomPrompts,
    dynamicColumns: {
      ...defaultPrompts.dynamicColumns,
      ...initialCustomPrompts.dynamicColumns
    }
  }));
  const [isUsingDefaults, setIsUsingDefaults] = useState(Object.keys(initialCustomPrompts).length === 0);
  const [promptSetName, setPromptSetName] = useState('');
  const [availablePromptSets, setAvailablePromptSets] = useState([]);

  useEffect(() => {
    fetchPromptSets();
  }, []);

  useEffect(() => {
    const dynamicColumnPrompts = { ...customPrompts.dynamicColumns };
    Object.entries(columnTypes).forEach(([column, type]) => {
      if (type === 'dynamic' && !dynamicColumnPrompts[column]) {
        dynamicColumnPrompts[column] = {
          system: `You are an AI assistant specializing in generating ${column} data.`,
          user: `Given the context and reference values, generate a suitable ${column} response:`
        };
      }
    });
    setCustomPrompts(prev => ({
      ...prev,
      dynamicColumns: dynamicColumnPrompts
    }));
  }, [columnTypes, customPrompts.dynamicColumns]);

  useEffect(() => {
    onSavePrompts(customPrompts);
  }, [customPrompts, onSavePrompts]);

  const fetchPromptSets = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/list_prompt_sets');
      setAvailablePromptSets(response.data);
    } catch (error) {
      console.error('Error fetching prompt sets:', error);
    }
  };

  const handlePromptChange = (category, field, value) => {
    setCustomPrompts(prev => ({
      ...prev,
      [category]: field ? { ...prev[category], [field]: value } : value
    }));
    setIsUsingDefaults(false);
  };

  const handleDynamicColumnPromptChange = (column, field, value) => {
    setCustomPrompts(prev => ({
      ...prev,
      dynamicColumns: {
        ...prev.dynamicColumns,
        [column]: { ...prev.dynamicColumns?.[column], [field]: value }
      }
    }));
    setIsUsingDefaults(false);
  };

  const handleSave = async () => {
    if (!promptSetName) {
      alert('Please enter a name for the prompt set');
      return;
    }
    try {
      await axios.post('http://localhost:5000/api/save_prompt_set', {
        name: promptSetName,
        prompts: customPrompts
      });
      alert('Prompt set saved successfully');
      fetchPromptSets();
      onSavePrompts(customPrompts);
    } catch (error) {
      console.error('Error saving prompt set:', error);
      alert('Error saving prompt set');
    }
  };

  const handleLoad = async (name) => {
    try {
      const response = await axios.get(`http://localhost:5000/api/load_prompt_set/${name}`);
      setCustomPrompts(response.data);
      setIsUsingDefaults(false);
      setPromptSetName(name);
      onSavePrompts(response.data);
    } catch (error) {
      console.error('Error loading prompt set:', error);
      alert('Error loading prompt set');
    }
  };

  const handleReset = () => {
    setCustomPrompts({ ...defaultPrompts });
    setIsUsingDefaults(true);
    setPromptSetName('');
    onSavePrompts({ ...defaultPrompts });
  };

  const renderPromptField = (label, value, onChange, gridProps = {}) => (
    <Grid item xs={12} sm={6} {...gridProps}>
      <TextField
        fullWidth
        multiline
        rows={8}
        label={label}
        value={value}
        onChange={onChange}
        variant="outlined"
        margin="normal"
        InputProps={{
          style: { fontFamily: 'monospace', fontSize: '0.9rem' }
        }}
      />
    </Grid>
  );

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>Custom Prompt Editor</Typography>
      <Typography variant="body2" gutterBottom color="textSecondary">
        {isUsingDefaults ? "Using default prompts. Edit to customize." : "Using custom prompts."}
      </Typography>
      
      <Box sx={{ mb: 2 }}>
        <TextField
          label="Prompt Set Name"
          value={promptSetName}
          onChange={(e) => setPromptSetName(e.target.value)}
          sx={{ mr: 2 }}
        />
        <Button variant="contained" color="primary" onClick={handleSave}>
          Save Prompt Set
        </Button>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Select
          value=""
          displayEmpty
          onChange={(e) => handleLoad(e.target.value)}
          sx={{ minWidth: 200, mr: 2 }}
        >
          <MenuItem value="" disabled>Load Prompt Set</MenuItem>
          {availablePromptSets.map(name => (
            <MenuItem key={name} value={name}>{name}</MenuItem>
          ))}
        </Select>
        <Button variant="outlined" color="secondary" onClick={handleReset}>
          Reset to Defaults
        </Button>
      </Box>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>Top-level System Prompt</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {renderPromptField(
              "Top-level System Prompt",
              customPrompts.topLevelSystem,
              (e) => handlePromptChange('topLevelSystem', null, e.target.value),
              { xs: 12 }
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>General System Prompt</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {renderPromptField(
              "System Prompt",
              customPrompts.system,
              (e) => handlePromptChange('system', null, e.target.value),
              { xs: 12 }
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>Paraphrase Prompts</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {renderPromptField(
              "Paraphrase System Prompt",
              customPrompts.paraphrase.system,
              (e) => handlePromptChange('paraphrase', 'system', e.target.value)
            )}
            {renderPromptField(
              "Paraphrase User Prompt",
              customPrompts.paraphrase.user,
              (e) => handlePromptChange('paraphrase', 'user', e.target.value)
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>Verification Prompts</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {renderPromptField(
              "Verification System Prompt",
              customPrompts.verify.system,
              (e) => handlePromptChange('verify', 'system', e.target.value)
            )}
            {renderPromptField(
              "Verification User Prompt",
              customPrompts.verify.user,
              (e) => handlePromptChange('verify', 'user', e.target.value)
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {Object.entries(columnTypes).filter(([_, type]) => type === 'dynamic').map(([column]) => (
        <Accordion key={column}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>{column} Prompts</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              {renderPromptField(
                `${column} System Prompt`,
                customPrompts.dynamicColumns[column]?.system || '',
                (e) => handleDynamicColumnPromptChange(column, 'system', e.target.value)
              )}
              {renderPromptField(
                `${column} User Prompt`,
                customPrompts.dynamicColumns[column]?.user || '',
                (e) => handleDynamicColumnPromptChange(column, 'user', e.target.value)
              )}
            </Grid>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
};

export default CustomPromptEditor;