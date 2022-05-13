import React from 'react';
import './App.css';
import {Report} from "../report/report";
import {Header} from "./header";
import {createTheme, ThemeProvider} from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#c6f1e7'
    },
    secondary: {
      main: '#59606d'
    }
  }
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <div className="App">
        <Header/>
        <Report/>
      </div>
    </ThemeProvider>
  );
}

export default App;
