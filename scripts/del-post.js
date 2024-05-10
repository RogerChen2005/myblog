/* This is a script to create a new post markdown file with front-matter */

import fs from "fs"
import path from "path"

const args = process.argv.slice(2)

if (args.length === 0) {
  console.error(`Error: No filename argument provided
Usage: npm run del -- <filename>`)
  process.exit(1) // Terminate the script and return error code 1
}

let fileName = args[0]

const targetDir = "./src/content/posts/";

let fullPath;
fullPath = path.join(targetDir, fileName+'.md');
if (fs.existsSync(fullPath)) {
  fs.rmSync(fullPath);
  console.log(`Post ${fullPath} deleted`);
}

fullPath = path.join(targetDir, fileName);
if (fs.existsSync(fullPath)) {
  fs.rmSync(fullPath,{
    'recursive':true,
    'force':true
  });
  console.log(`Post ${fullPath} deleted`);
}
