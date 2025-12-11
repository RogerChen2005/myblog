import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const postsDir = path.join(__dirname, '../src/content/posts')

function fixImagePaths(filePath) {
  let content = fs.readFileSync(filePath, 'utf8')
  let modified = false

  // Fix ![](attachments/...) -> ![](./attachments/...)
  const regex = /!\[([^\]]*)\]\(attachments\//g
  if (regex.test(content)) {
    content = content.replace(
      /!\[([^\]]*)\]\(attachments\//g,
      '![$1](./attachments/',
    )
    modified = true
  }

  if (modified) {
    fs.writeFileSync(filePath, content, 'utf8')
    console.log(`✓ Fixed: ${path.relative(postsDir, filePath)}`)
    return true
  }
  return false
}

function walkDir(dir) {
  const files = fs.readdirSync(dir)
  let fixedCount = 0

  for (const file of files) {
    const filePath = path.join(dir, file)
    const stat = fs.statSync(filePath)

    if (stat.isDirectory()) {
      fixedCount += walkDir(filePath)
    } else if (file.endsWith('.md')) {
      if (fixImagePaths(filePath)) {
        fixedCount++
      }
    }
  }

  return fixedCount
}

console.log('Scanning for image path issues...\n')
const fixed = walkDir(postsDir)
console.log(`\nFixed ${fixed} file(s).`)
