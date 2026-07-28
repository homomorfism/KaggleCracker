import { useRef, useState } from 'react'
import type { DragEvent, FormEvent } from 'react'
import { api, fmtBytes } from '../api'
import { navigate } from '../hooks'

export default function CreateProject() {
  const [name, setName] = useState('')
  const [target, setTarget] = useState('')
  const [description, setDescription] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return
    const picked = [...incoming].filter((f) => f.name.toLowerCase().endsWith('.csv'))
    if (picked.length < (incoming.length ?? 0)) setError('only .csv files are accepted')
    // Re-adding a file with the same name replaces it, like the server does.
    setFiles((prev) => [...prev.filter((p) => !picked.some((n) => n.name === p.name)), ...picked])
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    addFiles(e.dataTransfer.files)
  }

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    try {
      setBusy('creating project…')
      const project = await api.createProject({ name, description, target })
      for (const file of files) {
        setBusy(`uploading ${file.name}…`)
        await api.uploadFile(project.slug, file)
      }
      navigate(`/p/${project.slug}`)
    } catch (err) {
      setBusy(null)
      setError(String(err instanceof Error ? err.message : err))
    }
  }

  return (
    <section className="sheet">
      <div className="page-head">
        <h1>New project</h1>
        <a className="mono-dim" href="#/">◂ BACK</a>
      </div>

      <form className="form" onSubmit={submit}>
        <label className="field">
          <span className="field-label">PROJECT NAME</span>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Playground S6E7 — Student Health"
            autoFocus
          />
        </label>

        <label className="field">
          <span className="field-label">TARGET COLUMN <em>(what the model predicts)</em></span>
          <input
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            placeholder="health_condition"
          />
        </label>

        <label className="field">
          <span className="field-label">TASK DESCRIPTION</span>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={4}
            placeholder="What is being predicted, which metric, anything the agent should know…"
          />
        </label>

        <div
          className={`dropzone ${dragOver ? 'dropzone-over' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileInput.current?.click()}
        >
          <span className="dropzone-hint">
            {files.length ? 'DROP MORE CSV FILES OR CLICK' : 'DROP CSV FILES HERE OR CLICK TO PICK'}
          </span>
          <input
            ref={fileInput}
            type="file"
            accept=".csv"
            multiple
            hidden
            onChange={(e) => addFiles(e.target.files)}
          />
          {files.length > 0 && (
            <ul className="dropzone-files">
              {files.map((f) => (
                <li key={f.name} className="chip">
                  {f.name} <em>{fmtBytes(f.size)}</em>
                  <button
                    type="button"
                    className="chip-x"
                    aria-label={`remove ${f.name}`}
                    onClick={(e) => {
                      e.stopPropagation()
                      setFiles((prev) => prev.filter((p) => p.name !== f.name))
                    }}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {error && <p className="form-error">✗ {error}</p>}

        <div className="form-actions">
          <button className="btn btn-primary" disabled={!name.trim() || busy !== null}>
            {busy ?? 'CREATE PROJECT'}
          </button>
        </div>
      </form>
    </section>
  )
}
