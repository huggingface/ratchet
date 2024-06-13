import "./App.css";
import { Model, Quantization, default as init } from "@ratchet-ml/ratchet-web";
import { styled } from "@mui/material/styles";
import { useState, useEffect } from "react";
import {
  LinearProgress,
  TextField,
  Button,
  Container,
  Card,
  CardMedia,
  Stack,
  Box,
  Dialog,
  DialogActions,
  DialogContentText,
  DialogTitle,
  DialogContent,
  Typography,
  CardActions,
  InputAdornment,
  IconButton,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";

const VisuallyHiddenInput = styled("input")({
  clip: "rect(0 0 0 0)",
  clipPath: "inset(50%)",
  height: 1,
  overflow: "hidden",
  position: "absolute",
  bottom: 0,
  left: 0,
  whiteSpace: "nowrap",
  width: 1,
});

function App() {
  const [question, setQuestion] = useState("");
  const [generatedText, setGeneratedText] = useState("");
  const [image, setImage] = useState(new Uint8Array());
  const [progress, setProgress] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [accepted, setAccepted] = useState(false);
  const [isSupportedBrowser, setIsSupportedBrowser] = useState(true);
  const [ratchetDBExists, setRatchetDBExists] = useState(false);
  const [model, setModel] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    (async () => {
      await init();
      setRatchetDBExists(
        (await window.indexedDB.databases())
          .map((db) => db.name)
          .includes("ratchet"),
      );
      await setImage(
        new Uint8Array(
          await (
            await fetch(
              "https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg",
            )
          ).arrayBuffer(),
        ),
      );
    })();
  }, []);

  async function loadModel() {
    setAccepted(true);
    setProgress(2);
    setModel(
      await Model.load("Moondream", Quantization.Q8_0, (p) => setProgress(p)),
    );
    setProgress(100);
    setIsLoading(false);
  }

  async function runModel() {
    if (!model || isRunning) {
      return;
    }

    setGeneratedText("");

    let cb = (s) => {
      setGeneratedText((prevText) => {
        return prevText + s;
      });
    };

    setIsRunning(true);
    await model.run({ question: question, image_bytes: image, callback: cb });
    setIsRunning(false);
  }

  async function handleUpload(e) {
    if (e.target.files.length == 0) {
      return;
    }
    setImage(new Uint8Array(await e.target.files[0].arrayBuffer()));
  }

  async function keypress(e) {
    if (e.key === "Enter") {
      runModel();
      e.preventDefault();
    }
  }

  async function deleteWeights() {
    setAccepted(false);
    setProgress(0);
    setModel(null);
    await window.indexedDB.deleteDatabase("ratchet");
    setIsLoading(true);
  }

  return (
    <div className="App">
      <Container maxWidth="sm" sx={{ marginTop: "50px" }}>
        <Dialog
          open={isSupportedBrowser && !accepted}
          aria-labelledby="alert-dialog-title"
          aria-describedby="alert-dialog-description"
        >
          <DialogTitle id="alert-dialog-title">
            {navigator.gpu ? "Load Model" : "Unsupported Browser"}
          </DialogTitle>
          <DialogContent>
            <DialogContentText id="alert-dialog-description">
              {navigator.gpu
                ? "This app requires downloading a 2.2GB model which may take a few minutes. If the model has been previously downloaded, it will be loaded from cache."
                : "This app requires a browser that supports webgpu"}
            </DialogContentText>
          </DialogContent>
          {navigator.gpu ? (
            <DialogActions>
              <Button onClick={() => loadModel()} autoFocus>
                Load Model
              </Button>
            </DialogActions>
          ) : (
            <></>
          )}
        </Dialog>
        <Stack spacing={2}>
          <Box sx={{ justifyContent: "center", display: "flex" }}>
            <Typography>
              Moondream by{" "}
              <a href="https://github.com/vikhyat/moondream">Vikhyat</a> running
              on WebGpu via{" "}
              <a href="https://github.com/huggingface/ratchet">Ratchet</a>
            </Typography>
          </Box>
          <Box sx={{ justifyContent: "center", display: "flex" }}>
            <Card>
              <CardMedia
                sx={{ maxWidth: 377, maxHeight: 377 }}
                component="img"
                image={URL.createObjectURL(new Blob([image]))}
              />
              <CardActions sx={{ justifyContent: "center", display: "flex" }}>
                <Button
                  component="label"
                  role={undefined}
                  disabled={isLoading}
                  size="small"
                  variant="contained"
                >
                  Change Image
                  <VisuallyHiddenInput
                    type="file"
                    accept="image/png, image/jpeg"
                    onInput={handleUpload}
                  />
                </Button>
                <Button
                  component="label"
                  role={undefined}
                  variant="contained"
                  size="small"
                  disabled={isLoading}
                  sx={{backgroundColor: "#e57373"}}
                  onClick={() => deleteWeights()}
                >
                  Delete Weights
                </Button>
              </CardActions>
            </Card>
          </Box>
          <Box>
            <TextField
              fullWidth
              disabled={isLoading}
              label="Question"
              variant="outlined"
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={keypress}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton disabled={isLoading || isRunning}>
                      <SendIcon
                        color="primary"
                        disabled={isLoading || isRunning}
                        onClick={runModel}
                      />
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
          </Box>
          <div>
            <LinearProgress variant="determinate" value={progress} />
          </div>
          {isLoading && progress < 99 ? (
            <Box sx={{ justifyContent: "center", display: "flex" }}>
              <Typography>Downloading Weights...</Typography>
            </Box>
          ) : (
            <></>
          )}
          {isLoading && progress > 99 ? (
            <Box sx={{ justifyContent: "center", display: "flex" }}>
              <Typography>Preparing Weights...</Typography>
            </Box>
          ) : (
            <></>
          )}
          <div>
            <Typography>{generatedText}</Typography>
          </div>
        </Stack>
      </Container>
    </div>
  );
}

export default App;
