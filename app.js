const express = require('express');
const bodyParser = require('body-parser');

const { spawn } = require('child_process');

const app = express();
app.set('view engine', 'ejs');
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));


const executePython = async (script, args) => {
    const arguments = args.map(arg => arg.toString());

    const py = spawn("python", [script, ...arguments]);

    const result = await new Promise((resolve, reject) => {
        let output;

        // Get output from python script
        py.stdout.on('data', (data) => {
            output = JSON.parse(data);
        });

        // Handle erros
        py.stderr.on("data", (data) => {
            console.error(`[python] Error occured: ${data}`);
            reject(`Error occured in ${script}`);
        });

        py.on("exit", (code) => {
            console.log(`Child process exited with code ${code}`);
            resolve(output);
        });
    });
    return result;
}

app.get('/', async (req, res) => {
    res.render('index', {result: null});
});

app.post('/', async (req, res) => {
    const word = req.body.word;
    let result = null;
    if (!word) {
        result = {
            "Best Algorithm": ("0 s"),
            "Corrections": {"word": "No word provided"}
        }
        
        res.render('index',{ result: result });
    }
    try {
        result = await executePython('python/b.py', [word]);
        if(result.Corrections.word === undefined) {
            result.Corrections.word = "Word is true!";
        }
        res.render('index',{ result: result });
    } catch (error) {
        res.status(500).json({ error: error});
    }
});

app.listen(5000, () => {
    console.log('[server] Application started!')
});