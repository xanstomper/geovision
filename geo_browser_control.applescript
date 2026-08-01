-- GeoVision Browser Control via AppleScript
on run argv
    if (count of argv) < 1 then
        return "Usage: geo_browser_control.applescript <command> [args]"
    end if
    set cmd to item 1 of argv
    if cmd is "current_tab" then
        tell application "Google Chrome"
            if (count of windows) = 0 then return "No windows"
            set t to active tab of front window
            set url_str to URL of t
            set title_str to title of t
        end tell
        return url_str & "|||TITLE|||" & title_str
    end if
    if cmd is "open" then
        set url_str to item 2 of argv
        tell application "Google Chrome"
            make new tab at end of tabs of front window with properties {URL:url_str}
            activate
        end tell
        delay 4
        return "opened" & url_str
    end if
    if cmd is "screenshot" then
        set out_path to item 2 of argv
        do shell script "screencapture -x " & quoted form of out_path
        return out_path
    end if
    if cmd is "move_mouse" then
        set x to item 2 of argv
        set y to item 3 of argv
        tell application "System Events"
            set mouse location to {x, y}
        end tell
        return "moved"
    end if
    if cmd is "click" then
        set x to item 2 of argv
        set y to item 3 of argv
        tell application "System Events"
            click at {x, y}
        end tell
        return "clicked"
    end if
    if cmd is "type_text" then
        set txt to item 2 of argv
        tell application "System Events"
            keystroke txt
        end tell
        return "typed"
    end if
    return "Unknown command: " & cmd
end run
