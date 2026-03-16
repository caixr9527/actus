"use client"

import { useMemo, useState } from "react"
import { Check, ChevronsUpDown } from "lucide-react"
import type { SelectOption } from "@/lib/auth/options"
import { filterSelectOptions } from "@/lib/auth/options"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"

type SearchableSelectProps = {
  value: string
  options: SelectOption[]
  placeholder?: string
  searchPlaceholder?: string
  emptyText?: string
  disabled?: boolean
  className?: string
  onValueChange: (value: string) => void
}

export function SearchableSelect({
  value,
  options,
  placeholder = "请选择",
  searchPlaceholder = "请输入关键词搜索",
  emptyText = "无匹配项",
  disabled = false,
  className,
  onValueChange,
}: SearchableSelectProps) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState("")

  const selectedOption = useMemo(
    () => options.find((item) => item.value === value),
    [options, value],
  )
  const filteredOptions = useMemo(
    () => filterSelectOptions(options, query, value),
    [options, query, value],
  )

  return (
    <DropdownMenu
      open={open}
      onOpenChange={(nextOpen) => {
        setOpen(nextOpen)
        if (!nextOpen) {
          setQuery("")
        }
      }}
    >
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="outline"
          disabled={disabled}
          className={cn("w-full justify-between font-normal", className)}
        >
          <span className="truncate">
            {selectedOption?.label ?? placeholder}
          </span>
          <ChevronsUpDown className="size-4 opacity-60" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="start"
        className="w-[var(--radix-dropdown-menu-trigger-width)] p-2"
      >
        <Input
          autoFocus
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={searchPlaceholder}
        />
        <div className="mt-2 max-h-56 overflow-y-auto">
          {filteredOptions.length > 0 ? (
            <div className="flex flex-col gap-1">
              {filteredOptions.map((option) => {
                const isSelected = option.value === value
                return (
                  <button
                    key={option.value}
                    type="button"
                    className={cn(
                      "flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm",
                      "hover:bg-accent hover:text-accent-foreground",
                      isSelected ? "bg-accent text-accent-foreground" : "text-gray-700",
                    )}
                    onClick={() => {
                      onValueChange(option.value)
                      setOpen(false)
                    }}
                  >
                    <Check className={cn("size-4", isSelected ? "opacity-100" : "opacity-0")} />
                    <span className="truncate">{option.label}</span>
                  </button>
                )
              })}
            </div>
          ) : (
            <div className="px-2 py-2 text-xs text-gray-500">{emptyText}</div>
          )}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
